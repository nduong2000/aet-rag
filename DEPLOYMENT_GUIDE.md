# AET-RAG GCP Cloud Run Deployment Guide

This guide will help you set up automated deployment from GitHub to Google Cloud Run using Docker containers.

## 🚀 Quick Start

1. **Run the setup script:**
   ```bash
   chmod +x setup-gcp-deployment.sh
   ./setup-gcp-deployment.sh
   ```

2. **Add GitHub secrets** (provided by the setup script)

3. **Push to main branch** to trigger deployment

## 📋 Prerequisites

- Google Cloud Platform account with billing enabled
- GitHub repository with your code
- Docker installed locally
- Google Cloud SDK (`gcloud`) installed and authenticated

## 🔧 Manual Setup (Alternative to Script)

### 1. Install and Configure Google Cloud SDK

```bash
# Install gcloud CLI (if not already installed)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate and set project
gcloud auth login
gcloud config set project aethrag2
```

### 2. Enable Required APIs

```bash
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    iamcredentials.googleapis.com \
    cloudresourcemanager.googleapis.com \
    sts.googleapis.com \
    secretmanager.googleapis.com
```

### 3. Create Artifact Registry Repository

```bash
gcloud artifacts repositories create aet-rag-repo \
    --repository-format=docker \
    --location=us-east1 \
    --description="Docker repository for AET-RAG application"
```

### 4. Set Up Workload Identity Federation

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create github-pool \
    --location="global" \
    --display-name="GitHub Actions Pool"

# Create Workload Identity Provider
gcloud iam workload-identity-pools providers create-oidc github-provider \
    --workload-identity-pool=github-pool \
    --location="global" \
    --issuer-uri="https://token.actions.githubusercontent.com" \
    --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
    --attribute-condition="assertion.repository=='YOUR_GITHUB_USERNAME/aet-rag'"
```

### 5. Create Service Account

```bash
# Create service account
gcloud iam service-accounts create github-actions-sa \
    --display-name="GitHub Actions Service Account"

# Grant necessary roles
gcloud projects add-iam-policy-binding aethrag2 \
    --member="serviceAccount:github-actions-sa@aethrag2.iam.gserviceaccount.com" \
    --role="roles/run.developer"

gcloud projects add-iam-policy-binding aethrag2 \
    --member="serviceAccount:github-actions-sa@aethrag2.iam.gserviceaccount.com" \
    --role="roles/artifactregistry.writer"

gcloud projects add-iam-policy-binding aethrag2 \
    --member="serviceAccount:github-actions-sa@aethrag2.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"

# Allow GitHub Actions to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
    github-actions-sa@aethrag2.iam.gserviceaccount.com \
    --role="roles/iam.workloadIdentityUser" \
    --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/aet-rag"
```

### 6. Set Up Secret Manager

```bash
# Create secret for API key
gcloud secrets create gcp-service-account-key --data-file=api_key.json

# Grant access to the service account
gcloud secrets add-iam-policy-binding gcp-service-account-key \
    --member="serviceAccount:github-actions-sa@aethrag2.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 🔐 GitHub Repository Configuration

### Required Secrets

Add these secrets to your GitHub repository (Settings > Secrets and variables > Actions):

1. **WIF_PROVIDER**: `projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
2. **WIF_SERVICE_ACCOUNT**: `github-actions-sa@aethrag2.iam.gserviceaccount.com`

### Finding Your Project Number

```bash
gcloud projects describe aethrag2 --format='value(projectNumber)'
```

## 🐳 Docker Configuration

Your `Dockerfile` is already optimized for Cloud Run with:
- Port 8080 (Cloud Run standard)
- Gunicorn WSGI server
- Proper health checks
- Optimized for container startup

## 🚀 Deployment Process

### Automatic Deployment

The GitHub Action will automatically deploy when you:
1. Push to the `main` branch
2. Merge a pull request to `main`

### Manual Deployment

You can also deploy manually using gcloud:

```bash
# Build and push image
docker build -t us-east1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest .
docker push us-east1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest

# Deploy to Cloud Run
gcloud run deploy aet-rag-service \
    --image us-east1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest \
    --region us-east1 \
    --port 8080 \
    --memory 2Gi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --timeout 300 \
    --allow-unauthenticated
```

## 🔍 Monitoring and Debugging

### View Logs

```bash
# View Cloud Run logs
gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=aet-rag-service' --limit 50

# Follow logs in real-time
gcloud logging tail 'resource.type=cloud_run_revision AND resource.labels.service_name=aet-rag-service'
```

### Check Service Status

```bash
# Get service details
gcloud run services describe aet-rag-service --region us-east1

# List all revisions
gcloud run revisions list --service aet-rag-service --region us-east1
```

### Common Issues and Solutions

#### 1. Build Failures
- Check Dockerfile syntax
- Ensure all dependencies are in requirements-main.txt
- Verify file paths in COPY commands

#### 2. Authentication Issues
- Verify Workload Identity Federation setup
- Check GitHub secrets are correctly set
- Ensure service account has proper permissions

#### 3. Runtime Errors
- Check Cloud Run logs for startup errors
- Verify environment variables are set correctly
- Ensure port 8080 is properly exposed

#### 4. Memory/CPU Issues
- Increase memory allocation in the GitHub Action
- Optimize your application for container environments
- Consider using Cloud Run's CPU allocation options

## 🔧 Configuration Options

### Environment Variables

The deployment sets these environment variables:
- `GOOGLE_CLOUD_PROJECT`: Your GCP project ID
- `GCP_LOCATION`: Deployment region
- `PORT`: Container port (8080)

### Resource Limits

Current configuration:
- **Memory**: 2Gi
- **CPU**: 1 vCPU
- **Timeout**: 300 seconds
- **Concurrency**: 80 requests per instance
- **Min instances**: 0 (scales to zero)
- **Max instances**: 10

### Scaling Configuration

Modify the GitHub Action to adjust scaling:

```yaml
flags: |
  --port=8080
  --memory=4Gi          # Increase memory
  --cpu=2               # Increase CPU
  --min-instances=1     # Keep warm instances
  --max-instances=20    # Allow more scaling
  --concurrency=100     # More concurrent requests
```

## 🔒 Security Best Practices

1. **Use Workload Identity Federation** (not service account keys)
2. **Store secrets in Secret Manager**
3. **Limit IAM permissions** to minimum required
4. **Enable audit logging**
5. **Use private container registry**
6. **Implement proper authentication** in your app

## 📊 Cost Optimization

1. **Scale to zero** when not in use (min-instances=0)
2. **Right-size resources** based on actual usage
3. **Use efficient base images** (already using python:3.10-slim)
4. **Monitor usage** with Cloud Monitoring
5. **Set up billing alerts**

## 🚀 Advanced Features

### Blue-Green Deployments

```bash
# Deploy new revision without traffic
gcloud run deploy aet-rag-service \
    --image us-east1-docker.pkg.dev/aethrag2/aet-rag-repo/aet-rag-app:latest \
    --region us-east1 \
    --no-traffic

# Gradually shift traffic
gcloud run services update-traffic aet-rag-service \
    --to-revisions=REVISION-NAME=50 \
    --region us-east1
```

### Custom Domains

```bash
# Map custom domain
gcloud run domain-mappings create \
    --service aet-rag-service \
    --domain your-domain.com \
    --region us-east1
```

## 📞 Support

If you encounter issues:

1. Check the [GitHub Actions logs](https://github.com/YOUR_USERNAME/aet-rag/actions)
2. Review [Cloud Run documentation](https://cloud.google.com/run/docs)
3. Check [Workload Identity Federation guide](https://cloud.google.com/iam/docs/workload-identity-federation)
4. Use `gcloud` CLI for debugging

## 🎯 Next Steps

After successful deployment:

1. Set up monitoring and alerting
2. Configure custom domains
3. Implement CI/CD for different environments
4. Set up automated testing
5. Configure backup and disaster recovery 