# Cloud Build Auto-Deploy Setup Guide

## Step-by-Step Instructions

Based on the screenshot you're viewing, here's how to fill out the Cloud Build trigger form:

### 1. Repository Service
- ✅ Select: **"Developer Connect"** (2nd option)
- Click "CONNECT REPOSITORY" button

### 2. Connect GitHub Repository
- You'll be prompted to authenticate with GitHub
- Select repository: `donatdermaku/personal-investor-assistant`
- Click "Connect"

### 3. Repository Generation
- ✅ Select: **"2. Generation"** (recommended)

### 4. Configuration Section

**In the "Repository *" dropdown:**
- After connecting, select your repo from the dropdown

**In the "Zweig *" (Branch) field:**
- Enter: `^main$`
- This regular expression matches only the `main` branch exactly

**Configuration Type:**
- ✅ Keep selected: **"Cloud Build configuration file (YAML or JSON)"**

**Location:**
- ✅ Keep selected: **"Repository"**

**Location of the Cloud Build configuration file:**
- Enter: `/cloudbuild.yaml`
- ⚠️ Note: The file is already created in your repo root

### 5. Click "CREATE" at the bottom

## What Happens After Setup

✅ **Every time you merge to `main`:**
1. Cloud Build automatically triggers
2. Builds Docker image from your Dockerfile
3. Pushes to Google Container Registry
4. Deploys to Cloud Run service `personal-investor-assistant`
5. Takes ~2-3 minutes total

## Viewing Build Logs

After merging to main:
1. Go to Cloud Console → Cloud Build → History
2. Click on the running/completed build
3. See detailed logs for each step

## Testing the Setup

After creating the trigger:
1. Commit the `cloudbuild.yaml` file:
   ```bash
   git add cloudbuild.yaml CLOUD_BUILD_SETUP.md
   git commit -m "Add Cloud Build auto-deploy configuration"
   git push origin feature/refactoring-phase4
   ```

2. Merge Phase 4 PR to main

3. Watch Cloud Build → History for automatic deployment

## Debugging Failed Builds

If a build fails:
- Check Cloud Build → History → Click failed build
- Look for red error messages in logs
- Common issues:
  - Missing permissions (fix: IAM roles for Cloud Build service account)
  - Dockerfile errors (fix: test locally with `docker build .`)
  - Cloud Run timeout (already set to 300s in cloudbuild.yaml)

## Next Steps

After this trigger is set up and Phase 4 is merged:
- CSV uploads will work with all Phase 4 improvements
- Cache auto-retry active
- Admin refresh endpoints available
- All 117 tests verified
