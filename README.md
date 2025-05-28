# AET-RAG: Aetna Data Science Deep Research RAG System

A sophisticated Retrieval-Augmented Generation (RAG) system built with LangChain and LangGraph for expert knowledge extraction from Aetna Data Science documentation, including Universal, External Stop Loss, and Capitation Payment files.

## 🚀 Features

- **Deep Research RAG**: Multi-stage retrieval and analysis using LangGraph workflows
- **Expert Knowledge Engine**: Specialized for Aetna data science documentation
- **Hybrid Retrieval**: Combines semantic search, keyword matching, and field-specific retrieval
- **Citation System**: Provides detailed source citations with confidence scoring
- **Multiple AI Models**: Support for Gemini 2.5 Pro and Flash models
- **Web Interface**: Clean, modern UI for interactive queries
- **Auto-Deployment**: GitHub Actions integration for Cloud Run deployment

## 🏗️ Architecture

- **Backend**: Flask + LangChain + LangGraph
- **Vector Database**: ChromaDB for document embeddings
- **AI Models**: Google Vertex AI (Gemini 2.5 series)
- **Deployment**: Google Cloud Run with Workload Identity Federation
- **CI/CD**: GitHub Actions for automated deployment

## 🔐 Security & Authentication

This system uses **secure credential management** with different authentication methods for different environments:

### Local Development
- Uses `api_key.json` (service account key file) - **never committed to Git**
- Falls back to Application Default Credentials if key file not found
- Automatic project ID detection from credentials

### Cloud Run Production
- Uses **Workload Identity Federation** (no key files needed)
- Secure, keyless authentication via GitHub Actions
- Automatic credential management

### 🛡️ Security Features
- ✅ Credentials never exposed in Git repository
- ✅ Comprehensive `.gitignore` rules for sensitive files
- ✅ Template files for easy setup without exposing secrets
- ✅ Environment-specific authentication detection
- ✅ Secure CI/CD pipeline with Workload Identity Federation

## 📋 Quick Start

### 1. Local Development Setup

```bash
# Clone the repository
git clone https://github.com/nduong2000/aet-rag.git
cd aet-rag

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up credentials (choose one option):

# Option A: Create service account key (recommended)
gcloud iam service-accounts create local-dev-sa \
    --display-name="Local Development Service Account"

gcloud projects add-iam-policy-binding aethrag2 \
    --member="serviceAccount:local-dev-sa@aethrag2.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"

gcloud iam service-accounts keys create api_key.json \
    --iam-account=local-dev-sa@aethrag2.iam.gserviceaccount.com

# Option B: Use Application Default Credentials
gcloud auth application-default login
gcloud config set project aethrag2

# Test authentication
python test_auth.py

# Initialize database
python create_chroma_db.py

# Start the application
python main.py
```

Visit `http://localhost:8080` to access the web interface.

### 2. Cloud Deployment

The system automatically deploys to Google Cloud Run when you push to the `main` branch:

```bash
git add .
git commit -m "Your changes"
git push origin main
```

Monitor deployment at: https://github.com/nduong2000/aet-rag/actions

## 📚 Documentation

- **[LOCAL_SETUP.md](LOCAL_SETUP.md)** - Detailed local development setup
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Cloud deployment configuration
- **[api_key.json.template](api_key.json.template)** - Service account key template
- **[env.example](env.example)** - Environment configuration template

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and customize:

```env
# ChromaDB settings
CHROMA_DB_DIR=./chroma_db_data
COLLECTION_NAME=aetna_docs

# Model settings
EMBEDDING_MODEL=text-embedding-005
CHAT_MODEL=gemini-2.5-pro-preview-05-06

# GCP settings
GOOGLE_CLOUD_PROJECT=aethrag2
GCP_LOCATION=us-east1
```

### Available Models

- `gemini-2.5-pro-preview-05-06` - High accuracy, slower
- `gemini-2.5-flash-preview-04-17` - Faster responses, good accuracy

## 🧪 Testing

```bash
# Test authentication setup
python test_auth.py

# Test API endpoints
python test_api.py

# Test specific functionality
python test_citations.py
python test_embedding_fix.py
```

## 🛠️ Development

### Project Structure

```
aet-rag/
├── main.py                 # Main Flask application
├── create_chroma_db.py     # Database initialization
├── test_auth.py           # Authentication testing
├── requirements.txt       # Python dependencies
├── Dockerfile            # Container configuration
├── .github/workflows/    # CI/CD configuration
├── templates/           # Web UI templates
├── documents/          # Source documents
└── chroma_db_data/    # Vector database
```

### Key Components

- **ResearchState**: LangGraph state management for multi-stage processing
- **AetnaDataScienceRAGSystem**: Main RAG system class
- **Hybrid Retrieval**: Semantic + keyword + field-specific search
- **Citation Engine**: Source tracking and confidence assessment

## 🚀 Deployment Architecture

```
GitHub Repository
       ↓ (push to main)
GitHub Actions
       ↓ (Workload Identity Federation)
Google Cloud Build
       ↓ (Docker build & push)
Artifact Registry
       ↓ (deploy)
Cloud Run Service
```

## 🔍 Troubleshooting

### Common Issues

1. **Authentication Errors**
   ```bash
   python test_auth.py  # Diagnose auth issues
   gcloud auth list     # Check current auth
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Database Issues**
   ```bash
   rm -rf ./chroma_db_data
   python create_chroma_db.py
   ```

4. **Deployment Failures**
   - Check GitHub Actions logs
   - Verify GCP project settings
   - Ensure Workload Identity Federation is configured

### Debug Mode

```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
python main.py
```

## 📊 Performance

- **Response Time**: 2-5 seconds for complex queries
- **Accuracy**: 85-95% based on document coverage
- **Scalability**: Auto-scaling 0-10 instances on Cloud Run
- **Memory**: 2Gi per instance, optimized for large document processing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly (including `python test_auth.py`)
5. Submit a pull request

## 📄 License

This project is proprietary to Aetna Data Science team.

## 🆘 Support

For issues or questions:

1. Check the [troubleshooting section](#troubleshooting)
2. Review the [LOCAL_SETUP.md](LOCAL_SETUP.md) guide
3. Test authentication with `python test_auth.py`
4. Check application logs: `tail -f aetna_rag_system.log`

---

**🔒 Security Note**: This repository uses secure credential management. The `api_key.json` file is never committed to Git and is automatically ignored. See [LOCAL_SETUP.md](LOCAL_SETUP.md) for secure setup instructions. 