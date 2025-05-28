# Local Development Setup Guide

This guide explains how to set up the AET-RAG system for local development while keeping credentials secure.

## 🔐 Credential Management

The system uses different authentication methods for different environments:

- **Local Development**: Uses `api_key.json` (service account key file)
- **Cloud Run**: Uses Workload Identity Federation (no key files needed)

## 📋 Prerequisites

1. **Python 3.8+** installed
2. **Google Cloud SDK** installed and configured
3. **Git** installed
4. **GCP Project** with required APIs enabled

## 🚀 Quick Setup

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/nduong2000/aet-rag.git
cd aet-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Create Service Account for Local Development

```bash
# Set your project ID
export PROJECT_ID="aethrag2"  # Replace with your project ID

# Create a service account for local development
gcloud iam service-accounts create local-dev-sa \
    --display-name="Local Development Service Account" \
    --description="Service account for local development of AET-RAG"

# Grant necessary roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:local-dev-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:local-dev-sa@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.objectViewer"

# Create and download the key file
gcloud iam service-accounts keys create api_key.json \
    --iam-account=local-dev-sa@$PROJECT_ID.iam.gserviceaccount.com

echo "✅ Service account key created: api_key.json"
```

### 3. Alternative: Use Application Default Credentials

If you prefer not to use a service account key file:

```bash
# Authenticate with your user account
gcloud auth application-default login

# Set the project
gcloud config set project aethrag2
```

**Note**: If using this method, comment out or remove the `api_key.json` logic in `main.py`.

### 4. Environment Configuration

Create a `.env` file for local configuration:

```bash
# Copy the template
cp env.example .env

# Edit the .env file with your settings
# The system will automatically detect and use appropriate credentials
```

### 5. Initialize ChromaDB

```bash
# Run the database initialization script
python create_chroma_db.py
```

### 6. Start the Application

```bash
# Run the Flask application
python main.py
```

The application will be available at `http://localhost:8080`

## 🔧 Configuration Options

### Environment Variables

Create a `.env` file with these optional settings:

```env
# ChromaDB settings
CHROMA_DB_DIR=./chroma_db_data
COLLECTION_NAME=aetna_docs

# Model settings
EMBEDDING_MODEL=text-embedding-005
CHAT_MODEL=gemini-2.5-pro-preview-05-06

# Retrieval settings
MAX_RETRIEVAL_DOCS=50
SIMILARITY_THRESHOLD=0.3
CHUNK_SIZE=1500
CHUNK_OVERLAP=200

# GCP settings (usually auto-detected)
GOOGLE_CLOUD_PROJECT=aethrag2
GCP_LOCATION=us-east1
```

## 🔍 Authentication Flow

The application automatically detects the environment and uses appropriate authentication:

### Local Development
1. Checks if `api_key.json` exists
2. If found, uses it for authentication
3. If not found, falls back to Application Default Credentials
4. Extracts project ID from the key file or environment

### Cloud Run
1. Detects Cloud Run environment (`K_SERVICE` environment variable)
2. Uses Workload Identity Federation automatically
3. Gets project ID from `GOOGLE_CLOUD_PROJECT` environment variable

## 🛠️ Troubleshooting

### Common Issues

#### 1. "api_key.json not found"
```bash
# Solution 1: Create the service account key (recommended)
gcloud iam service-accounts keys create api_key.json \
    --iam-account=local-dev-sa@aethrag2.iam.gserviceaccount.com

# Solution 2: Use application default credentials
gcloud auth application-default login
```

#### 2. "Permission denied" errors
```bash
# Check your authentication
gcloud auth list

# Check your project
gcloud config get-value project

# Re-authenticate if needed
gcloud auth application-default login
```

#### 3. "Project ID not found"
```bash
# Set the project explicitly
export GOOGLE_CLOUD_PROJECT="aethrag2"

# Or add it to your .env file
echo "GOOGLE_CLOUD_PROJECT=aethrag2" >> .env
```

#### 4. ChromaDB initialization fails
```bash
# Remove existing database and recreate
rm -rf ./chroma_db_data
python create_chroma_db.py
```

### Debug Mode

Enable debug logging by setting:

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
```

## 🔒 Security Best Practices

### ✅ Do's
- Keep `api_key.json` in `.gitignore` (already configured)
- Use service accounts with minimal required permissions
- Rotate service account keys regularly
- Use Application Default Credentials when possible

### ❌ Don'ts
- Never commit `api_key.json` to version control
- Don't share service account keys
- Don't use overly permissive IAM roles
- Don't hardcode credentials in code

## 🚀 Deployment

When you're ready to deploy:

1. **Commit your changes** (credentials are automatically excluded):
   ```bash
   git add .
   git commit -m "Your changes"
   git push origin main
   ```

2. **Automatic deployment** will trigger via GitHub Actions to Cloud Run

3. **Monitor the deployment**:
   ```bash
   # Check GitHub Actions
   # Visit: https://github.com/nduong2000/aet-rag/actions
   
   # Check Cloud Run logs
   gcloud logging read 'resource.type=cloud_run_revision AND resource.labels.service_name=aet-rag-service' --limit 50
   ```

## 📚 Additional Resources

- [Google Cloud Authentication Guide](https://cloud.google.com/docs/authentication)
- [Workload Identity Federation](https://cloud.google.com/iam/docs/workload-identity-federation)
- [Cloud Run Documentation](https://cloud.google.com/run/docs)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## 🆘 Need Help?

If you encounter issues:

1. Check the application logs: `tail -f aetna_rag_system.log`
2. Verify your GCP setup: `gcloud auth list && gcloud config list`
3. Test authentication: `gcloud auth application-default print-access-token`
4. Check the [troubleshooting section](#troubleshooting) above 