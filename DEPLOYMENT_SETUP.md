# AET-RAG Deployment Setup Guide

## ✅ Completed Setup

### 1. Artifact Registry Repository
- **Repository Name**: `aet-rag-repo`
- **Location**: `us-central1`
- **Format**: Docker
- **Registry URI**: `us-central1-docker.pkg.dev/aethrag2/aet-rag-repo`
- **Status**: ✅ Created and verified

### 2. Service Account Permissions

#### rag-service-account@aethrag2.iam.gserviceaccount.com
- ✅ `roles/aiplatform.admin` - For Vertex AI models
- ✅ `roles/aiplatform.user` - For Vertex AI access
- ✅ `roles/artifactregistry.writer` - For pushing Docker images
- ✅ `roles/run.developer` - For Cloud Run deployment

#### github-actions-sa@aethrag2.iam.gserviceaccount.com (GitHub Actions)
- ✅ `roles/artifactregistry.writer` - For pushing Docker images
- ✅ `roles/run.developer` - For Cloud Run deployment
- ✅ `roles/storage.admin` - For storage access
- ✅ `roles/iam.serviceAccountUser` - For service account impersonation

### 3. GitHub Actions Workflow
- **File**: `.github/workflows/google-cloudrun-docker.yml`
- **Trigger**: Push to main branch or merged PR
- **Region**: `us-central1` (supports all Gemini models)
- **Service**: `aet-rag-service`
- **Status**: ✅ Configured and ready

### 4. Docker Configuration
- ✅ Docker configured for `us-central1-docker.pkg.dev`
- ✅ gcloud credential helper registered

## 🚀 Deployment Process

### Automatic Deployment
1. Push code to `main` branch or merge a PR
2. GitHub Actions will:
   - Authenticate using Workload Identity Federation
   - Build Docker image
   - Push to Artifact Registry
   - Deploy to Cloud Run
   - Output service URL

### Manual Deployment (if needed)
```bash
# Build and tag image
docker build -t us-central1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest .

# Push to Artifact Registry
docker push us-central1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest

# Deploy to Cloud Run
gcloud run deploy aet-rag-service \
  --image us-central1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest \
  --region us-central1 \
  --port 8080 \
  --memory 2Gi \
  --cpu 1 \
  --min-instances 0 \
  --max-instances 10 \
  --timeout 300 \
  --allow-unauthenticated \
  --set-env-vars GOOGLE_CLOUD_PROJECT=aethrag2,GCP_LOCATION=us-central1
```

## 🔧 Configuration Details

### Environment Variables (Cloud Run)
- `GOOGLE_CLOUD_PROJECT`: `aethrag2`
- `GCP_LOCATION`: `us-central1`

### Available Models (us-central1)
- ✅ `gemini-2.5-flash-preview-04-17` (default)
- ✅ `gemini-2.5-pro-preview-05-06`
- ✅ `gemini-2.0-flash-001`
- ✅ `gemini-2.0-flash-lite-001`
- ✅ `gemini-1.5-flash-001`
- ✅ `gemini-1.5-pro-001`

### Resource Limits
- **Memory**: 2Gi
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests
- **Scaling**: 0-10 instances

## 🔍 Monitoring & Troubleshooting

### View Logs
```bash
# View Cloud Run logs
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=aet-rag-service' --limit 50 --format json

# View GitHub Actions logs
# Go to: https://github.com/nduong2000/aet-rag/actions
```

### Health Check
- **Endpoint**: `https://[service-url]/health`
- **Expected Response**: JSON with system status

### Common Issues
1. **Permission Denied**: Check service account permissions
2. **Model Not Available**: Verify region supports the model
3. **Build Failures**: Check Dockerfile and dependencies
4. **Authentication Issues**: Verify Workload Identity Federation setup

## 📋 Next Steps

1. **Test Deployment**: Push a commit to trigger deployment
2. **Verify Service**: Check health endpoint after deployment
3. **Monitor Performance**: Watch logs and metrics
4. **Scale as Needed**: Adjust resource limits based on usage

## 🔐 Security Notes

- ✅ `api_key.json` excluded from Docker image
- ✅ Workload Identity Federation for secure authentication
- ✅ Service account with minimal required permissions
- ✅ No hardcoded credentials in code or workflows

---

**Last Updated**: 2025-05-28
**Status**: Ready for deployment 