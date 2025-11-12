# Cloud Deployment Guide

Complete guide to deploy Canvas Extension Backend to production with cloud storage.

---

## 🎯 Overview

The backend now supports **two modes**:

| Mode | Storage | Database | Use Case |
|------|---------|----------|----------|
| **Local** (default) | `./uploads` folder | SQLite (`./data/chats.db`) | Development, testing |
| **Cloud** (production) | Google Cloud Storage | PostgreSQL | Production, scaling |

**The app automatically detects which mode to use based on environment variables** - no code changes needed!

---

## 📦 What Changed

### New Files Created
1. **`utils/storage_manager.py`** - Google Cloud Storage integration
2. **`migrate_to_cloud.py`** - Migration script for existing data
3. **`CLOUD_DEPLOYMENT.md`** - This guide

### Modified Files
1. **`requirements.txt`** - Added cloud dependencies
2. **`.env.example`** - Added cloud configuration
3. **`.gitignore`** - Added cloud credentials
4. **`server.py`** - Cloud storage support with local fallback
5. **`utils/chat_storage.py`** - PostgreSQL support with SQLite fallback
6. **`utils/document_manager.py`** - GCS support with local fallback

---

## 🚀 Quick Start (Cloud Deployment)

### Prerequisites
- Google Cloud Platform account
- GCP project created
- Basic knowledge of GCP Console

### Step 1: Set Up Google Cloud Storage

```bash
# 1. Create GCS bucket
gsutil mb -p YOUR_PROJECT_ID -l us-central1 gs://canvas-extension-pdfs

# 2. Enable Cloud Storage API (if not enabled)
gcloud services enable storage-api.googleapis.com

# 3. Create service account
gcloud iam service-accounts create canvas-extension-sa \
    --display-name="Canvas Extension Service Account"

# 4. Grant storage admin permissions
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:canvas-extension-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"

# 5. Download credentials JSON
gcloud iam service-accounts keys create gcp-service-account.json \
    --iam-account=canvas-extension-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com
```

### Step 2: Set Up PostgreSQL Database

**Option A: Supabase (Free Tier - Recommended for starting)**
1. Go to [supabase.com](https://supabase.com)
2. Create new project
3. Get connection string from Settings → Database
4. Format: `postgresql://postgres:password@db.xxx.supabase.co:5432/postgres`

**Option B: Google Cloud SQL**
```bash
# Create PostgreSQL instance
gcloud sql instances create canvas-extension-db \
    --database-version=POSTGRES_15 \
    --tier=db-f1-micro \
    --region=us-central1

# Create database
gcloud sql databases create canvas_extension \
    --instance=canvas-extension-db

# Create user
gcloud sql users create canvas_user \
    --instance=canvas-extension-db \
    --password=YOUR_SECURE_PASSWORD

# Get connection string
gcloud sql instances describe canvas-extension-db
```

### Step 3: Configure Environment

Update your `.env` file:

```bash
# Copy example
cp .env.example .env

# Edit .env with your values
nano .env
```

Required values:
```env
# GCS Configuration
GCS_BUCKET_NAME=canvas-extension-pdfs
GCS_PROJECT_ID=your-gcp-project-id
GOOGLE_APPLICATION_CREDENTIALS=./gcp-service-account.json

# PostgreSQL (use Supabase or Cloud SQL connection string)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Existing config (keep these)
GOOGLE_API_KEY=your-gemini-api-key
```

### Step 4: Install Dependencies

```bash
# Install new cloud dependencies
pip install -r requirements.txt
```

### Step 5: Migrate Existing Data (if you have data)

```bash
# Run migration script
python migrate_to_cloud.py
```

This will:
- Upload all PDFs from `./uploads/` to GCS
- Export SQLite chat history to PostgreSQL
- Preserve all metadata

### Step 6: Test Cloud Storage

```bash
# Start server
python server.py
```

You should see:
```
✅ Storage Manager (GCS) initialized
✅ Document Manager initialized
✅ Chat Storage (PostgreSQL) initialized
```

If you see "falling back to local storage", check your `.env` configuration.

---

## 🔧 Deployment Options

### Option 1: Railway (Easiest)

1. **Push code to GitHub**
2. **Connect Railway to repo**
3. **Add environment variables**:
   - All variables from `.env`
   - Upload `gcp-service-account.json` as file or encode as base64
4. **Deploy** (automatic)

### Option 2: Render

1. **Create Web Service**
2. **Connect GitHub repo**
3. **Configure**:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python server.py`
4. **Add environment variables**
5. **Deploy**

### Option 3: Google Cloud Run (Advanced)

```bash
# Build and deploy
gcloud run deploy canvas-extension-backend \
    --source . \
    --region us-central1 \
    --allow-unauthenticated \
    --set-env-vars GCS_BUCKET_NAME=canvas-extension-pdfs,GCS_PROJECT_ID=your-project-id
```

### Option 4: DigitalOcean App Platform

1. **Create new app from GitHub**
2. **Configure environment**
3. **Add environment variables**
4. **Deploy**

---

## 💰 Cost Estimate

### Google Cloud Storage
- Storage: $0.02/GB/month
- Egress (downloads): $0.12/GB
- Operations: $0.004 per 10,000 operations

**Example (100 PDFs, 200MB total):**
- Storage: $0.004/month
- Downloads (100 downloads/month): $0.024/month
- **Total: ~$0.03/month**

### PostgreSQL Database

| Option | Cost | Limits |
|--------|------|--------|
| **Supabase Free** | $0/month | 500MB storage, 2GB bandwidth |
| **Cloud SQL (f1-micro)** | $10/month | Shared core, 3GB storage |
| **Supabase Pro** | $25/month | 8GB storage, 50GB bandwidth |

### Total Monthly Cost
- **Free tier**: $0 (Supabase) + $0.03 (GCS) = **$0.03/month**
- **Paid tier**: $10-25/month

---

## 🔒 Security Best Practices

### 1. Protect Credentials
```bash
# NEVER commit these files:
gcp-service-account.json
.env
*.key
*-credentials.json
```

### 2. GCS Bucket Security
```bash
# Make bucket private (default)
gsutil iam ch allUsers:objectViewer gs://canvas-extension-pdfs
# Remove above if you added it accidentally

# Use signed URLs for temporary access
# (Already implemented in storage_manager.py)
```

### 3. Database Security
- Use strong passwords
- Enable SSL connections
- Whitelist IP addresses
- Rotate credentials regularly

### 4. Environment Variables
Never hardcode:
- API keys
- Database passwords
- Service account credentials

Always use environment variables or secret managers.

---

## 🧪 Testing

### Test GCS Upload
```python
python3 -c "
from utils.storage_manager import StorageManager
import os

sm = StorageManager(
    bucket_name=os.getenv('GCS_BUCKET_NAME'),
    project_id=os.getenv('GCS_PROJECT_ID')
)

# Upload test file
with open('test.pdf', 'rb') as f:
    blob_name = sm.upload_pdf('test_course', 'test.pdf', f.read())
    print(f'Uploaded: {blob_name}')

# List files
files = sm.list_files('test_course')
print(f'Files: {files}')

# Delete test
sm.delete_file(blob_name)
print('Test passed!')
"
```

### Test PostgreSQL Connection
```python
python3 -c "
from utils.chat_storage import ChatStorage
import os

cs = ChatStorage(database_url=os.getenv('DATABASE_URL'))
cs.save_chat_session('test_session', 'test_course', [
    {'role': 'user', 'content': 'Hello'},
    {'role': 'assistant', 'content': 'Hi there!'}
])
print('Chat saved successfully!')

chats = cs.get_recent_chats('test_course')
print(f'Found {len(chats)} chats')
"
```

---

## 🐛 Troubleshooting

### "GCS not configured, falling back to local storage"
**Cause**: Missing environment variables

**Fix**:
```bash
# Check .env has these set:
GCS_BUCKET_NAME=your-bucket-name
GCS_PROJECT_ID=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=./gcp-service-account.json

# Verify JSON file exists
ls -la gcp-service-account.json
```

### "Failed to initialize PostgreSQL"
**Cause**: Invalid DATABASE_URL

**Fix**:
```bash
# Test connection
psql "postgresql://user:pass@host:5432/dbname"

# If it works in psql but not in app, check:
# - No extra spaces in DATABASE_URL
# - Correct password (no special characters issues)
# - Database exists
```

### "Permission denied" on GCS
**Cause**: Service account lacks permissions

**Fix**:
```bash
# Grant storage admin role
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID \
    --member="serviceAccount:YOUR_SA@YOUR_PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

### Migration Script Fails
**Cause**: Missing dependencies or permissions

**Fix**:
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt

# Check file permissions
ls -la ./uploads
ls -la ./data/chats.db

# Run with verbose output
python migrate_to_cloud.py 2>&1 | tee migration.log
```

---

## 📊 Monitoring

### Check Storage Usage
```bash
# GCS bucket size
gsutil du -sh gs://canvas-extension-pdfs

# PostgreSQL size
psql -c "SELECT pg_size_pretty(pg_database_size('canvas_extension'));"
```

### Application Logs
```bash
# Railway/Render
# Check dashboard logs

# Google Cloud Run
gcloud run services logs read canvas-extension-backend

# Local
tail -f server.log
```

---

## 🔄 Rollback Plan

If cloud deployment fails, rollback is automatic:

1. **Remove cloud env vars from `.env`**:
   ```bash
   # Comment out these lines:
   # GCS_BUCKET_NAME=...
   # DATABASE_URL=...
   ```

2. **Restart server** - automatically falls back to local storage

3. **Your local data is safe** - `./uploads/` and `./data/chats.db` unchanged

---

## 📚 Additional Resources

- [Google Cloud Storage Docs](https://cloud.google.com/storage/docs)
- [Supabase Docs](https://supabase.com/docs)
- [Railway Docs](https://docs.railway.app)
- [Render Docs](https://render.com/docs)

---

## ✅ Deployment Checklist

- [ ] GCP project created
- [ ] GCS bucket created and configured
- [ ] Service account created with credentials JSON
- [ ] PostgreSQL database created (Supabase or Cloud SQL)
- [ ] `.env` file configured with all cloud credentials
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Existing data migrated (`python migrate_to_cloud.py`)
- [ ] Server tested locally with cloud storage
- [ ] Credentials added to .gitignore
- [ ] Code pushed to GitHub
- [ ] Deployment platform configured (Railway/Render/etc)
- [ ] Environment variables set on platform
- [ ] Application deployed and tested
- [ ] Local backups kept until verified working

---

## 🎉 Success!

Once deployed, your backend will:
- ✅ Store PDFs in Google Cloud Storage (unlimited, scalable)
- ✅ Store chat history in PostgreSQL (ACID-compliant, backed up)
- ✅ Work from anywhere (no local dependencies)
- ✅ Scale horizontally (multiple instances)
- ✅ Cost ~$0-10/month (depending on usage)

Questions? Check the troubleshooting section or create an issue on GitHub.
