# Test scanPHY with 1 ESP32-S3 + a Windows laptop

With one board, scanPHY can do **presence** (someone is here or not), **activity** (empty / still / moving)
and **breathing rate**. Position and zones need 3 boards, so you can ignore the x/y values in this setup.

Time needed: about 30 minutes.

---

## 0. What you need

- An ESP32-S3 dev board (for example ESP32-S3-DevKitC-1) and a **data** USB-C cable. Some cables only charge.
- The Windows laptop and the board on the **same 2.4 GHz WiFi**. A normal home router is fine. Turn off
  "AP/client isolation" if your router has it.
- **Python 3.12** (3.10 or 3.11 also work) from https://www.python.org/downloads/. During setup, tick **"Add python.exe to PATH"**.
  Avoid 3.13/3.14 and the free-threaded `3.14t` build: some packages, PyTorch especially, may not have wheels for them.
- Git from https://git-scm.com/download/win. Or download the repo as a ZIP from GitHub
  (switch to branch `claude/ruview-esp32-jetson-alternative-qodw4d`, then **Code → Download ZIP**).

## 1. Get the code (PowerShell)

```powershell
cd $HOME
git clone -b claude/ruview-esp32-jetson-alternative-qodw4d https://github.com/Anurag123rk/scanPHY.git
cd scanPHY
py -3.12 -m venv .venv          # `py -0p` lists installed versions
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-tools.txt
.\.venv\Scripts\python.exe -m pip install torch      # only needed for training (step 9); CPU build is fine
```

Every command below calls `.\.venv\Scripts\python.exe` directly, so you never have to "activate" the venv.
That avoids PowerShell execution-policy errors.

Check the install by running the demo:

```powershell
.\.venv\Scripts\python.exe -m scanphy serve --source sim
```

Open http://localhost:3000. You should see a simulated person walking around. Press **Ctrl+C** to stop it.

## 2. Find your laptop's IP address

```powershell
ipconfig
```

Under **Wireless LAN adapter Wi-Fi**, note the **IPv4 Address**, for example `192.168.1.23`. The board will send
its data to this address.

> Tip: laptop IPs can change after a reboot. If the board stops showing up later, check this again, or give the
> laptop a fixed IP (DHCP reservation) in your router.

## 3. Open the firewall (PowerShell **as Administrator**, once)

```powershell
New-NetFirewallRule -DisplayName "scanPHY CSI UDP 5005" -Direction Inbound -Protocol UDP -LocalPort 5005 -Action Allow -Profile Any
New-NetFirewallRule -DisplayName "scanPHY dashboard 3000-3001" -Direction Inbound -Protocol TCP -LocalPort 3000,3001 -Action Allow -Profile Any
```

If Windows later shows a "Windows Defender Firewall has blocked some features" pop-up for Python, click **Allow**.

## 4. Connect the board and find its COM port

1. Plug the S3 in. Many DevKitC boards have **two** USB-C ports. Either works:
   - **UART / COM**: the USB-serial chip. You may need the driver: CP210x from silabs.com or CH343 from wch-ic.com.
   - **USB**: the native USB port. No driver needed on Windows 10/11.
2. Open **Device Manager → Ports (COM & LPT)**. You should see something like `USB-SERIAL CH343 (COM5)` or
   `USB JTAG/serial debug unit (COM7)`. Note the COM number.

## 5. Flash the firmware and settings (one command)

Back in a normal PowerShell, in the `scanPHY` folder:

```powershell
.\.venv\Scripts\python.exe tools\provision.py --port COM5 --node-id 1 `
    --ssid "YOUR_WIFI_NAME" --password "YOUR_WIFI_PASSWORD" `
    --target-ip 192.168.1.23 --flash-firmware
```

- Replace `COM5`, the WiFi name/password and the IP with your own values. The backticks `` ` `` continue the
  command on the next line.
- The prebuilt firmware in `firmware\release\` is used, so you don't need to install ESP-IDF.
- **If it says "Failed to connect":** hold the **BOOT** button, tap **RESET/EN**, release **BOOT**, and run the
  command again.
- When it finishes, press **RESET/EN** once to start the new firmware.

## 6. Check that the board connects (optional but useful)

```powershell
.\.venv\Scripts\python.exe -m serial.tools.miniterm COM5 115200
```

Within a few seconds you should see:

```
I scanphy: node 1 -> 192.168.1.23:5005, ssid 'YOUR_WIFI_NAME', 50 Hz
I scanphy: connected to aa:bb:cc:dd:ee:ff ch 6 rssi -48
I scanphy: pinging gateway 192.168.1.1 every 20 ms
I scanphy: csi cb 251, sent 250, dropped 0, heap ...
```

Press **Ctrl+]** to exit. If you see `disconnected, retrying` over and over, the WiFi name or password is wrong,
or the network is 5 GHz-only. The ESP32-S3 needs **2.4 GHz**.

## 7. Check that data reaches the laptop

```powershell
.\.venv\Scripts\python.exe -m scanphy probe
```

Expected output:

```
node   1  192.168.1.40     48.0 frames/s  rssi  -47 dBm  64 subcarriers
```

If you see "no CSI frames received", go back to step 3 (firewall) and check the `--target-ip` you used in step 5.

## 8. Run scanPHY with a one-board room

Place the board about **1 m high**, **2–4 m from the router**, so that you can walk through the line between
the router and the board. That line is where a single board senses best.

```powershell
copy config\room.single-node.example.json config\room.json
notepad config\room.json
```

Edit `width`/`depth` (room size in metres), plus the router and node `x`/`y` positions measured from one floor
corner. Save the file, then:

```powershell
.\.venv\Scripts\python.exe -m scanphy serve
```

Open **http://localhost:3000**. NODE-1 should show **STREAMING** at about 40–50 frames/s, with a moving CSI waterfall.

### First test

1. **Calibrate tab → Capture baseline.** Leave the room, or at least stand far away and keep still, for 20 s.
2. Come back and walk between the router and the board. **Occupancy** should switch to **MOVING**/**PRESENT**,
   and the motion graph should jump.
3. Sit still **on or near the line between the router and the board**, about 1–2 m from the board, for 30 s or more.
   **Vitals → Breathing** should show a number, typically 12–20 bpm.
   With a single board, someone sitting still *away* from that line can read as **EMPTY**: one link barely
   sees them. Moving people are detected almost anywhere. Step 9 and more boards both help.
4. Leave again. After a few seconds it should say **EMPTY**.

## 9. Teach it your room (training on the laptop)

The pretrained model was trained on simulated rooms, so a short calibration with your real room helps a lot.
On the **Calibrate** tab, use **manual activity** with **RECORD (NO POSITION)**:

| Record | Seconds | How many | What to do |
|---|---|---|---|
| `empty`  | 30 | 3× | nobody in the room (start it, then walk out) |
| `still`  | 30 | 3× | sit or stand still, in a different spot each time |
| `moving` | 30 | 3× | walk around normally |

Then go to **Train → TRAIN NOW**. Leave "use my recordings" ticked. On a laptop CPU this takes a few minutes.
You can watch the learning curve while it runs. When it finishes, the new model becomes active automatically,
and the top bar shows `MODEL ROOM-…`.

The trained model is a single file, `models\room-YYYYMMDD-HHMMSS.npz`. Copy it to the Jetson later, or upload it
under **Models → Upload .npz**.

### Teach it to count people

People counting needs examples from *your* room. Open the **People** tab:
1. Set **max people in this room** (for example 2).
2. Click each situation you can act out: *Nobody*, *1 person still*, *1 person walking*, *2 people…*.
   You get a 5 s countdown to get into position, then it records 30 s. Do each one 2–3 times.
3. Click **TRAIN WITH MY RECORDINGS**. This takes a few minutes on a laptop CPU; progress shows on the **Train** tab.
   The new model is switched on automatically when it finishes.

With one board, counting beyond one person stays rough even after teaching. Two or three boards help a lot.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `provision.py` can't find the COM port | Use a data cable, install the CP210x/CH343 driver, or try the other USB-C port |
| "Failed to connect to ESP32-S3" | Hold BOOT, tap RESET, release BOOT, then retry |
| Serial log loops on `disconnected, retrying` | Wrong WiFi password, or the network is 5 GHz-only |
| Board connects but `probe` shows nothing | Firewall (step 3), wrong `--target-ip`, or VPN on the laptop |
| `probe` says UDP 5005 is busy | `scanphy serve` is already running; just use the dashboard |
| Always "MOVING" even in an empty room | Capture the baseline again with the room really empty, then do step 9 |
| Breathing shows `···` | It needs about 13 s of you being still before it shows a number |
| `Unable to create process using ...python3.14t.exe` | The `py` launcher points to a removed Python. Install 3.12 and use `py -3.12 -m venv .venv` |
| `The module '.venv' could not be loaded` | The venv wasn't created (see the row above), or you left out the `.\` prefix |
| Laptop IP changed | Re-run step 5 without `--flash-firmware` (this only rewrites the settings) |
