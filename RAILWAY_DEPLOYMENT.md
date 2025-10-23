# Railway Deployment Guide

Quick guide to deploy your Canvas Extension Backend to Railway.

---

## ✅ Prerequisites

- [x] GitHub repo: `https://github.com/RushilRandhar/CanvasExt-backend.git`
- [x] `Procfile` created
- [x] `railway.json` created
- [x] server.py configured for Railway
- [ ] Railway account (sign up at railway.app)

---

## 🚀 Step-by-Step Deployment

### Step 1: Push Latest Changes to GitHub

```bash
cd "/Users/rushilrandhar/Canvas Extension/canvas-extension-backend"

# Add new files
git add Procfile railway.json server.py

# Commit
git commit -m "Add Railway deployment configuration"

# Push
git push origin main
```

### Step 2: Deploy to Railway

1. **Go to Railway**: https://railway.app

2. **Sign in with GitHub**

3. **Create New Project**:
   - Click **"New Project"**
   - Select **"Deploy from GitHub repo"**
   - Choose **"RushilRandhar/CanvasExt-backend"**
   - Railway will automatically start building

4. **Wait for Initial Build** (~2 minutes)
   - Railway detects Python from requirements.txt
   - Installs dependencies
   - First deploy may show errors (that's OK - we need to add env vars)

### Step 3: Configure Environment Variables

Click on your deployed service, then go to **"Variables"** tab:

#### Required Variables:

```bash
GOOGLE_API_KEY=<your-gemini-api-key-from-.env>
HOST=0.0.0.0
PORT=${{PORT}}
```

#### Optional (for Cloud Storage):

If using Google Cloud Storage:

```bash
GCS_BUCKET_NAME=canvas-extension-pdfs
GCS_PROJECT_ID=<your-gcp-project-id>
```

**For GCP credentials, choose ONE method:**

**Method A: Base64 Encoded (Easiest)**
```bash
# On your local machine:
cd canvas-extension-backend
base64 -i gcp-service-account.json | tr -d '\n' > gcp-base64.txt

# Copy the entire contents of gcp-base64.txt
```

Then in Railway Variables tab:
```bash
GCP_SERVICE_ACCOUNT_BASE64=<paste-entire-base64-string-here>
```

**Method B: Skip GCS for Now**
Leave GCS variables empty. Backend will automatically fall back to local storage (SQLite).
You can add GCS later when ready for production.

#### Optional (for PostgreSQL):

If using PostgreSQL instead of SQLite:

```bash
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

**To use Railway's PostgreSQL:**
1. In Railway project dashboard, click **"+ New"**
2. Select **"Database" → "PostgreSQL"**
3. Railway creates database and provides `DATABASE_URL`
4. Copy `DATABASE_URL` to your service's variables

**Or use Supabase (free tier):**
- Sign up at supabase.com
- Create project
- Get connection string from Settings → Database
- Add as `DATABASE_URL`

### Step 4: Redeploy

After adding variables:
1. Click **"Redeploy"** button or just wait (auto-redeploys)
2. Check **"Deployments"** tab for logs
3. Look for success messages:
   ```
   ✅ Document Manager initialized
   ✅ Root Agent initialized
   ✅ Backend ready!
   ```

### Step 5: Generate Public Domain

1. In your service, go to **"Settings"** tab
2. Scroll to **"Networking"** section
3. Click **"Generate Domain"**
4. Copy your URL (e.g., `https://canvasext-backend-production.up.railway.app`)

### Step 6: Test Your Backend

```bash
# Replace with your Railway URL
curl https://your-app.up.railway.app/

# Expected response:
{"status":"ok","message":"AI Study Assistant Backend is running"}
```

### Step 7: Update Chrome Extension Frontend

Update backend URL in your Chrome extension:

**File: `canvas-extension-frontend/chat/websocket-client.js`**
```javascript
// Around line 3-5, change:
this.backendUrl = 'http://localhost:8000';

// To:
this.backendUrl = 'https://your-app.up.railway.app';
```

**File: `canvas-extension-frontend/popup/popup-v2.js`**
```javascript
// Find BackendClient initialization (around line 620):
const backendClient = new BackendClient('http://localhost:8000');

// Change to:
const backendClient = new BackendClient('https://your-app.up.railway.app');
```

**Don't forget to update WebSocket URL too:**
```javascript
// Should use wss:// (not ws://) for HTTPS
wsClient.connect('wss://your-app.up.railway.app/ws/chat/' + courseId);
```

### Step 8: Reload Extension and Test

1. Go to `chrome://extensions`
2. Click reload icon on your extension
3. Test:
   - Upload PDFs
   - Start a chat
   - Verify WebSocket connection works
   - Check chat history saves

---

## 🔍 Troubleshooting

### Build Fails

**Check Railway logs** (Deployments tab → Click on deployment):

Common issues:
- **Missing dependencies**: Make sure requirements.txt is committed
- **Python version**: Railway uses Python 3.11 (should work fine)

Fix:
```bash
# Ensure requirements.txt is up to date
git add requirements.txt
git commit -m "Update requirements"
git push
```

### App Crashes on Startup

**Check Runtime Logs**:

Common issues:
- Missing `GOOGLE_API_KEY` → Add in Variables tab
- Port binding issue → Make sure `PORT=${{PORT}}` is set

### "GCS not configured" Warning

This is **normal** if you haven't added GCS credentials yet.

The app will automatically:
- Use local filesystem storage (./uploads)
- Use SQLite for chat history (./data/chats.db)

This works for development/testing. Add GCS later for production.

### WebSocket Connection Fails

**From Chrome Extension:**

Make sure you're using:
- `https://` for REST API calls
- `wss://` for WebSocket connections (not `ws://`)

Railway automatically provides HTTPS, so always use secure protocols.

### CORS Errors

Check server.py has:
```python
allow_origins=["chrome-extension://*", "*"]
```

This should already be configured.

### Can't Upload PDFs

**Check:**
1. Railway logs show "Backend ready"
2. `/upload_pdfs` endpoint is accessible
3. You added `GOOGLE_API_KEY` variable

---

## 💰 Railway Pricing

### Free Tier
- **$5 credit/month** (resets monthly)
- ~500 hours of runtime
- Perfect for development and testing
- No credit card required

### Hobby Plan ($5/month)
- Recommended for production
- Always-on instances (no sleep)
- Better performance
- Remove usage limits

**To Monitor Usage:**
- Railway Dashboard → **"Usage"** tab
- Set up email alerts at 80% usage

---

## 📊 Optional: Add PostgreSQL Database

If you want Railway-hosted PostgreSQL:

1. In Railway project, click **"+ New"**
2. Select **"Database" → "PostgreSQL"**
3. Railway creates database instantly
4. Copy `DATABASE_URL` from database settings
5. Add to your service's Variables
6. Redeploy

**Cost:** $5/month for 1GB storage

**Alternative (Free):** Use Supabase free tier (500MB)

---

## 🔒 Security Checklist

- [x] `.env` file is gitignored ✓
- [x] `gcp-service-account.json` is gitignored ✓
- [ ] Strong database password (if using PostgreSQL)
- [ ] GCP service account has minimal permissions
- [ ] Railway environment variables are set
- [ ] HTTPS enabled (automatic on Railway)

---

## 🔄 Automatic Deployments

Railway automatically redeploys when you push to GitHub!

```bash
# Make any changes
git add .
git commit -m "Update feature"
git push origin main

# Railway automatically:
# 1. Detects push
# 2. Builds new version
# 3. Runs tests (if configured)
# 4. Deploys
# 5. Switches traffic to new version
```

**No manual redeployment needed!**

---

## 📈 Monitoring

### View Logs
- Railway Dashboard → Your service → **"Deployments"** tab
- Click any deployment to see logs
- Real-time log streaming available

### Check Metrics
- Dashboard shows:
  - CPU usage
  - Memory usage
  - Request count
  - Response times

### Set Up Alerts
- Settings → Notifications
- Email alerts for:
  - Deployment failures
  - High resource usage
  - Downtime

---

## 🎉 You're Done!

Your backend is now:
- ✅ Live at `https://your-app.up.railway.app`
- ✅ Auto-deploys on git push
- ✅ HTTPS enabled
- ✅ WebSocket support enabled
- ✅ Logs available in dashboard
- ✅ Scales automatically

### Next Steps:

1. **Test thoroughly** with your Chrome extension
2. **Monitor first few days** for any issues
3. **Add PostgreSQL** when ready for production
4. **Set up GCS** for scalable file storage
5. **Upgrade to Hobby plan** ($5/month) when leaving development

---

## 🆘 Need Help?

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Check Logs**: Railway Dashboard → Deployments → Click deployment

Common questions:
- WebSocket issues? Make sure using `wss://` not `ws://`
- CORS errors? Check `allow_origins` in server.py
- Can't access backend? Check domain is generated in Settings → Networking

---

## 📝 Quick Reference

### Your URLs:
- **Backend**: `https://your-app.up.railway.app`
- **Health Check**: `https://your-app.up.railway.app/`
- **Upload PDFs**: `https://your-app.up.railway.app/upload_pdfs`
- **WebSocket**: `wss://your-app.up.railway.app/ws/chat/{course_id}`

### Important Commands:
```bash
# View logs
railway logs

# Deploy manually
railway up

# Link to existing project
railway link

# View variables
railway variables
```

(Install Railway CLI: `npm install -g @railway/cli` or `brew install railway`)

---

Happy deploying! 🚀
