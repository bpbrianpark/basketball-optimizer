# Basketball Shot Form Optimizer – Mobile App

Expo (React Native) app for the Basketball Shot Form Optimizer. Record or pick a short clip, then analyze shooting form via the backend or (later) on-device ML.

## Install

From this directory (`mobile/`):

```bash
npm install
```

## Run

```bash
npx expo start
```

Then:

- **iOS simulator:** press `i` in the terminal, or scan the QR code with your iPhone (Expo Go app).
- **Android emulator:** press `a`, or scan with the Expo Go app on Android.
- **Physical device:** Install [Expo Go](https://expo.dev/go), ensure your phone and computer are on the same Wi‑Fi, and scan the QR code.

## Backend URL (for “Analyze on my computer”)

To analyze videos on your own machine instead of on the device:

1. From the **project root** (parent of `mobile/` and `backend/`), start the FastAPI backend:
   ```bash
   uvicorn backend.app.main:app --reload --host 0.0.0.0
   ```
2. In the app, open **Settings** and set **Backend base URL** to your computer’s LAN address and port, e.g. `http://192.168.1.100:8000`. (Use `localhost` only when using a simulator on the same machine.)
3. The URL is saved automatically when you tap outside the field. The API client (used when MOB-6 is implemented) will use this URL for upload and analyze requests.

## Tabs

- **Home** – Welcome and link to Capture.
- **Capture** – Placeholder for record/pick video (MOB-3, MOB-4).
- **Results** – Placeholder for score and overlay (MOB-6, MOB-7).
- **Settings** – Backend base URL (persisted with AsyncStorage).

## Next steps (see MOBILE_APP_TASKS.md)

- MOB-3: Camera – record short clip.
- MOB-4: Gallery – pick existing video.
- MOB-6: Real API client – upload, analyze, get result (and overlay when backend exposes it).
