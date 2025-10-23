# Quick Deploy to Railway - TL;DR

## 1️⃣ Push to GitHub (5 min)

```bash
cd "/Users/rushilrandhar/Canvas Extension/canvas-extension-backend"

# Add Railway files
git add Procfile railway.json RAILWAY_DEPLOYMENT.md server.py

# Commit
git commit -m "Add Railway deployment configuration"

# Push
git push origin main
```

## 2️⃣ Deploy on Railway (10 min)

1. **Go to**: https://railway.app
2. **Sign in** with GitHub
3. **New Project** → **Deploy from GitHub repo**
4. **Select**: `RushilRandhar/CanvasExt-backend`
5. **Wait** ~2 minutes for initial build

## 3️⃣ Add Environment Variables

In Railway dashboard → **Variables** tab:

### Required (Copy from your local `.env`):
```bash
GOOGLE_API_KEY=<your-gemini-api-key>
HOST=0.0.0.0
PORT=${{PORT}}
```

### Optional - For GCS (if you want cloud storage):
```bash
GCS_BUCKET_NAME=canvas-ext-docs
GCS_PROJECT_ID=gen-lang-client-0366448563
```

**For GCS credentials**, run this on your Mac:
```bash
cd "/Users/rushilrandhar/Canvas Extension/canvas-extension-backend"
base64 -i gcp-service-account.json | tr -d '\n' > gcp-base64.txt
cat gcp-base64.txt
```

Then add in Railway:
```bash
GCP_SERVICE_ACCOUNT_BASE64=<paste-the-base64-output-here>
```

### Optional - For PostgreSQL:
**Skip for now** - backend will use SQLite automatically

## 4️⃣ Generate Public Domain

Railway dashboard → **Settings** → **Networking** → **Generate Domain**

Copy your URL: `https://your-app.up.railway.app`

## 5️⃣ Test Backend

```bash
curl https://your-app.up.railway.app/
```

Should return:
```json
{"status":"ok","message":"AI Study Assistant Backend is running"}
```

## 6️⃣ Update Chrome Extension

**File**: `canvas-extension-frontend/chat/websocket-client.js`

Change line 3:
```javascript
this.backendUrl = 'https://your-app.up.railway.app';  // Replace with your Railway URL
```

**File**: `canvas-extension-frontend/popup/popup-v2.js`

Change around line 620:
```javascript
const backendClient = new BackendClient('https://your-app.up.railway.app');
```

**WebSocket URL** (chat.js around line 551):
```javascript
wsClient.connect('wss://your-app.up.railway.app/ws/chat/' + courseId);  // Use wss:// not ws://
```

## 7️⃣ Reload Extension

1. Go to `chrome://extensions`
2. Click reload icon on your extension
3. Test uploading PDFs and chatting

---

## ✅ Done!

Your backend is now live at Railway with:
- HTTPS + WebSocket support
- Auto-deployments on git push
- Free tier ($5/month credit)

## 🆘 Issues?

Check Railway logs: Dashboard → **Deployments** → Click deployment

Common fixes:
- Missing `GOOGLE_API_KEY` → Add in Variables
- WebSocket errors → Use `wss://` not `ws://`
- Can't upload PDFs → Check logs for errors

Full guide: See [RAILWAY_DEPLOYMENT.md](RAILWAY_DEPLOYMENT.md)
