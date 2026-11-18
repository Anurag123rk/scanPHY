/*
 * WaveSense 3D node firmware for ESP32-S3.
 *
 * Joins the WiFi network, pings the router at a fixed rate so the router keeps
 * transmitting, captures the Channel State Information (CSI) of every frame received
 * from the router and streams it to the WaveSense 3D server (Jetson) as UDP datagrams.
 *
 * Configuration (SSID, password, server IP/port, node id, rate) is read from NVS
 * namespace "wavesense" (written by tools/provision.py) and falls back to the
 * menuconfig defaults.
 *
 * Wire format: see wavesense/protocol.py (24-byte header + int8 [imag, real] pairs).
 */

#include <string.h>
#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_netif.h"
#include "esp_mac.h"
#include "nvs_flash.h"
#include "nvs.h"
#include "lwip/sockets.h"
#include "lwip/inet.h"
#include "ping/ping_sock.h"

static const char *TAG = "wavesense";

#define WAVESENSE_MAGIC   0x59485053u   /* "SPHY" */
#define WAVESENSE_VERSION 1
#define HEADER_SIZE     24
#define MAX_CSI_BYTES   384
#define QUEUE_DEPTH     24

typedef struct {
    char ssid[33];
    char pass[65];
    char ip[16];
    uint16_t port;
    uint8_t node;
    uint16_t rate;
} node_cfg_t;

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint8_t version;
    uint8_t node_id;
    uint8_t n_antennas;
    uint8_t flags;
    uint16_t n_subcarriers;
    uint16_t channel;
    uint32_t seq;
    uint32_t timestamp_ms;
    int8_t rssi;
    int8_t noise_floor;
    uint16_t reserved;
} frame_header_t;

_Static_assert(sizeof(frame_header_t) == HEADER_SIZE, "header must be 24 bytes");

typedef struct {
    uint16_t len;
    uint8_t data[HEADER_SIZE + MAX_CSI_BYTES];
} packet_t;

static node_cfg_t s_cfg;
static QueueHandle_t s_queue;
static uint8_t s_bssid[6];
static volatile bool s_connected = false;
static uint32_t s_seq = 0;
static volatile uint32_t s_sent = 0, s_dropped = 0, s_cb = 0;
static int64_t s_last_us = 0;
static esp_ping_handle_t s_ping = NULL;

/* ------------------------------------------------------------------ configuration */
static void load_config(void)
{
    memset(&s_cfg, 0, sizeof(s_cfg));
    strlcpy(s_cfg.ssid, CONFIG_WAVESENSE_WIFI_SSID, sizeof(s_cfg.ssid));
    strlcpy(s_cfg.pass, CONFIG_WAVESENSE_WIFI_PASSWORD, sizeof(s_cfg.pass));
    strlcpy(s_cfg.ip, CONFIG_WAVESENSE_TARGET_IP, sizeof(s_cfg.ip));
    s_cfg.port = CONFIG_WAVESENSE_TARGET_PORT;
    s_cfg.node = CONFIG_WAVESENSE_NODE_ID;
    s_cfg.rate = CONFIG_WAVESENSE_RATE_HZ;

    nvs_handle_t h;
    if (nvs_open("wavesense", NVS_READONLY, &h) == ESP_OK) {
        size_t n;
        n = sizeof(s_cfg.ssid); nvs_get_str(h, "ssid", s_cfg.ssid, &n);
        n = sizeof(s_cfg.pass); nvs_get_str(h, "pass", s_cfg.pass, &n);
        n = sizeof(s_cfg.ip);   nvs_get_str(h, "ip", s_cfg.ip, &n);
        nvs_get_u16(h, "port", &s_cfg.port);
        nvs_get_u8(h, "node", &s_cfg.node);
        nvs_get_u16(h, "rate", &s_cfg.rate);
        nvs_close(h);
        ESP_LOGI(TAG, "configuration loaded from NVS");
    } else {
        ESP_LOGW(TAG, "no provisioning found, using menuconfig defaults");
    }
    if (s_cfg.rate < 5) s_cfg.rate = 5;
    if (s_cfg.rate > 200) s_cfg.rate = 200;
    ESP_LOGI(TAG, "node %u -> %s:%u, ssid '%s', %u Hz", s_cfg.node, s_cfg.ip, s_cfg.port, s_cfg.ssid, s_cfg.rate);
}

/* ------------------------------------------------------------------ CSI capture */
static void csi_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || info->len <= 0 || !s_connected) return;
    /* Only use frames transmitted by our access point: a fixed, known link. */
    if (memcmp(info->mac, s_bssid, 6) != 0) return;
    s_cb++;

    /* Rate limit to ~1.5x the target rate. */
    int64_t now = esp_timer_get_time();
    int64_t min_gap = 1000000LL / (s_cfg.rate + s_cfg.rate / 2);
    if (now - s_last_us < min_gap) return;
    s_last_us = now;

    packet_t pkt;
    uint16_t len = info->len > MAX_CSI_BYTES ? MAX_CSI_BYTES : info->len;
    frame_header_t *h = (frame_header_t *)pkt.data;
    h->magic = WAVESENSE_MAGIC;
    h->version = WAVESENSE_VERSION;
    h->node_id = s_cfg.node;
    h->n_antennas = 1;
    h->flags = 0;
    h->n_subcarriers = len / 2;
    h->channel = info->rx_ctrl.channel;
    h->seq = s_seq++;
    h->timestamp_ms = (uint32_t)(now / 1000);
    h->rssi = info->rx_ctrl.rssi;
    h->noise_floor = info->rx_ctrl.noise_floor;
    h->reserved = 0;
    memcpy(pkt.data + HEADER_SIZE, info->buf, len);
    if (info->first_word_invalid) {
        memset(pkt.data + HEADER_SIZE, 0, 4 < len ? 4 : len);
        h->flags |= 0x01;
    }
    pkt.len = HEADER_SIZE + len;
    if (xQueueSend(s_queue, &pkt, 0) != pdTRUE) s_dropped++;
}

static void csi_init(void)
{
    wifi_csi_config_t csi = {
        .lltf_en = true,
        .htltf_en = false,
        .stbc_htltf2_en = false,
        .ltf_merge_en = true,
        .channel_filter_en = false,
        .manu_scale = false,
        .shift = 0,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

/* ------------------------------------------------------------------ UDP sender */
static void sender_task(void *arg)
{
    struct sockaddr_in dst = {0};
    dst.sin_family = AF_INET;
    dst.sin_port = htons(s_cfg.port);
    dst.sin_addr.s_addr = inet_addr(s_cfg.ip);
    int sock = -1;
    packet_t pkt;
    int64_t last_log = esp_timer_get_time();
    /* lwIP must be up before any socket call: wait until WiFi has an IP address. */
    while (!s_connected) vTaskDelay(pdMS_TO_TICKS(100));
    for (;;) {
        if (sock < 0) {
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            if (sock < 0) { vTaskDelay(pdMS_TO_TICKS(500)); continue; }
        }
        if (xQueueReceive(s_queue, &pkt, pdMS_TO_TICKS(1000)) == pdTRUE) {
            int r = sendto(sock, pkt.data, pkt.len, 0, (struct sockaddr *)&dst, sizeof(dst));
            if (r < 0) { s_dropped++; } else { s_sent++; }
        }
        int64_t now = esp_timer_get_time();
        if (now - last_log > 5000000) {
            ESP_LOGI(TAG, "csi cb %lu, sent %lu, dropped %lu, heap %lu",
                     (unsigned long)s_cb, (unsigned long)s_sent, (unsigned long)s_dropped,
                     (unsigned long)esp_get_free_heap_size());
            last_log = now;
        }
    }
}

/* ------------------------------------------------------------------ traffic generator */
static void start_ping(const esp_ip4_addr_t *gw)
{
    if (s_ping) {
        esp_ping_stop(s_ping);
        esp_ping_delete_session(s_ping);
        s_ping = NULL;
    }
    esp_ping_config_t pc = ESP_PING_DEFAULT_CONFIG();
    pc.target_addr.type = IPADDR_TYPE_V4;
    pc.target_addr.u_addr.ip4.addr = gw->addr;
    pc.count = ESP_PING_COUNT_INFINITE;
    pc.interval_ms = 1000 / s_cfg.rate;
    pc.timeout_ms = 1000;
    pc.data_size = 8;
    pc.task_stack_size = 3072;
    esp_ping_callbacks_t cbs = {0};
    if (esp_ping_new_session(&pc, &cbs, &s_ping) == ESP_OK) {
        esp_ping_start(s_ping);
        ESP_LOGI(TAG, "pinging gateway " IPSTR " every %lu ms", IP2STR(gw), (unsigned long)pc.interval_ms);
    }
}

/* ------------------------------------------------------------------ WiFi */
static void wifi_event(void *arg, esp_event_base_t base, int32_t id, void *data)
{
    if (base == WIFI_EVENT && id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (base == WIFI_EVENT && id == WIFI_EVENT_STA_DISCONNECTED) {
        s_connected = false;
        ESP_LOGW(TAG, "disconnected, retrying");
        vTaskDelay(pdMS_TO_TICKS(1000));
        esp_wifi_connect();
    } else if (base == IP_EVENT && id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *ev = (ip_event_got_ip_t *)data;
        wifi_ap_record_t ap;
        if (esp_wifi_sta_get_ap_info(&ap) == ESP_OK) {
            memcpy(s_bssid, ap.bssid, 6);
            ESP_LOGI(TAG, "connected to " MACSTR " ch %d rssi %d", MAC2STR(ap.bssid), ap.primary, ap.rssi);
        }
        ESP_LOGI(TAG, "ip " IPSTR " gw " IPSTR, IP2STR(&ev->ip_info.ip), IP2STR(&ev->ip_info.gw));
        s_connected = true;
        start_ping(&ev->ip_info.gw);
    }
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, wifi_event, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    wifi_config_t wc = {0};
    strlcpy((char *)wc.sta.ssid, s_cfg.ssid, sizeof(wc.sta.ssid));
    strlcpy((char *)wc.sta.password, s_cfg.pass, sizeof(wc.sta.password));
    wc.sta.threshold.authmode = strlen(s_cfg.pass) ? WIFI_AUTH_WPA_PSK : WIFI_AUTH_OPEN;
    wc.sta.pmf_cfg.capable = true;
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wc));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));  /* no power save: steady CSI */
    csi_init();
}

void app_main(void)
{
    esp_err_t err = nvs_flash_init();
    if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        err = nvs_flash_init();
    }
    ESP_ERROR_CHECK(err);
    ESP_LOGI(TAG, "WaveSense 3D node firmware starting");
    load_config();
    s_queue = xQueueCreate(QUEUE_DEPTH, sizeof(packet_t));
    wifi_init();  /* initialises esp_netif / lwIP first */
    xTaskCreatePinnedToCore(sender_task, "wavesense_tx", 4096, NULL, 5, NULL, 1);
}
