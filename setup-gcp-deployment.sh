#!/bin/bash

# AET-RAG GCP Cloud Run Deployment Setup Script
# This script sets up everything needed for automated deployment to GCP Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="aethrag2"
REGION="us-east1"
SERVICE_NAME="aet-rag-service"
REPOSITORY_NAME="aet-rag-repo"
WIF_POOL_NAME="github-pool"
WIF_PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT_NAME="github-actions-sa"
GITHUB_REPO="nduong2000/aet-rag"  # Updated to correct GitHub repo

echo -e "${BLUE}🚀 Setting up GCP Cloud Run deployment for AET-RAG${NC}"
echo -e "${BLUE}Project ID: ${PROJECT_ID}${NC}"
echo -e "${BLUE}Region: ${REGION}${NC}"
echo ""

# Function to check if user is logged in to gcloud
check_gcloud_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        echo -e "${RED}❌ You are not logged in to gcloud. Please run 'gcloud auth login' first.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ gcloud authentication verified${NC}"
}

# Function to set the project
set_project() {
    echo -e "${YELLOW}📋 Setting GCP project...${NC}"
    gcloud config set project $PROJECT_ID
    echo -e "${GREEN}✓ Project set to ${PROJECT_ID}${NC}"
}

# Function to enable required APIs
enable_apis() {
    echo -e "${YELLOW}🔧 Enabling required Google Cloud APIs...${NC}"
    
    apis=(
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
        "artifactregistry.googleapis.com"
        "iamcredentials.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "sts.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        echo "Enabling $api..."
        gcloud services enable $api
    done
    
    echo -e "${GREEN}✓ All required APIs enabled${NC}"
}

# Function to create Artifact Registry repository
create_artifact_registry() {
    echo -e "${YELLOW}📦 Creating Artifact Registry repository...${NC}"
    
    if gcloud artifacts repositories describe $REPOSITORY_NAME --location=$REGION &>/dev/null; then
        echo -e "${GREEN}✓ Artifact Registry repository already exists${NC}"
    else
        gcloud artifacts repositories create $REPOSITORY_NAME \
            --repository-format=docker \
            --location=$REGION \
            --description="Docker repository for AET-RAG application"
        echo -e "${GREEN}✓ Artifact Registry repository created${NC}"
    fi
}

# Function to create service account for GitHub Actions
create_service_account() {
    echo -e "${YELLOW}👤 Creating service account for GitHub Actions...${NC}"
    
    if gcloud iam service-accounts describe "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" &>/dev/null; then
        echo -e "${GREEN}✓ Service account already exists${NC}"
    else
        gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
            --display-name="GitHub Actions Service Account" \
            --description="Service account for GitHub Actions to deploy to Cloud Run"
        echo -e "${GREEN}✓ Service account created${NC}"
        
        # Wait for service account to propagate
        echo "Waiting for service account to propagate..."
        sleep 10
    fi
    
    # Grant necessary roles with retry logic
    echo "Granting IAM roles..."
    roles=(
        "roles/run.developer"
        "roles/artifactregistry.writer"
        "roles/iam.serviceAccountUser"
        "roles/storage.admin"
    )
    
    for role in "${roles[@]}"; do
        echo "Granting role: $role"
        retry_count=0
        max_retries=3
        
        while [ $retry_count -lt $max_retries ]; do
            if gcloud projects add-iam-policy-binding $PROJECT_ID \
                --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
                --role="$role" &>/dev/null; then
                echo "✓ Successfully granted $role"
                break
            else
                retry_count=$((retry_count + 1))
                if [ $retry_count -lt $max_retries ]; then
                    echo "Retrying in 5 seconds... (attempt $retry_count/$max_retries)"
                    sleep 5
                else
                    echo -e "${RED}❌ Failed to grant $role after $max_retries attempts${NC}"
                    echo "You may need to grant this role manually:"
                    echo "gcloud projects add-iam-policy-binding $PROJECT_ID --member=\"serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com\" --role=\"$role\""
                fi
            fi
        done
    done
    
    echo -e "${GREEN}✓ IAM roles granted${NC}"
}

# Function to set up Workload Identity Federation
setup_workload_identity() {
    echo -e "${YELLOW}🔐 Setting up Workload Identity Federation...${NC}"
    
    # Create Workload Identity Pool
    if gcloud iam workload-identity-pools describe $WIF_POOL_NAME --location="global" &>/dev/null; then
        echo -e "${GREEN}✓ Workload Identity Pool already exists${NC}"
    else
        gcloud iam workload-identity-pools create $WIF_POOL_NAME \
            --location="global" \
            --display-name="GitHub Actions Pool"
        echo -e "${GREEN}✓ Workload Identity Pool created${NC}"
    fi
    
    # Create Workload Identity Provider
    if gcloud iam workload-identity-pools providers describe $WIF_PROVIDER_NAME \
        --workload-identity-pool=$WIF_POOL_NAME --location="global" &>/dev/null; then
        echo -e "${GREEN}✓ Workload Identity Provider already exists${NC}"
    else
        gcloud iam workload-identity-pools providers create-oidc $WIF_PROVIDER_NAME \
            --workload-identity-pool=$WIF_POOL_NAME \
            --location="global" \
            --issuer-uri="https://token.actions.githubusercontent.com" \
            --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
            --attribute-condition="assertion.repository=='${GITHUB_REPO}'"
        echo -e "${GREEN}✓ Workload Identity Provider created${NC}"
    fi
    
    # Allow GitHub Actions to impersonate the service account with retry logic
    echo "Setting up Workload Identity binding..."
    retry_count=0
    max_retries=3
    
    while [ $retry_count -lt $max_retries ]; do
        if gcloud iam service-accounts add-iam-policy-binding \
            "${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
            --role="roles/iam.workloadIdentityUser" \
            --member="principalSet://iam.googleapis.com/projects/$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${WIF_POOL_NAME}/attribute.repository/${GITHUB_REPO}" &>/dev/null; then
            echo -e "${GREEN}✓ Workload Identity Federation configured${NC}"
            break
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Retrying Workload Identity binding in 5 seconds... (attempt $retry_count/$max_retries)"
                sleep 5
            else
                echo -e "${RED}❌ Failed to configure Workload Identity binding after $max_retries attempts${NC}"
                echo "You may need to run this command manually:"
                echo "gcloud iam service-accounts add-iam-policy-binding ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com --role=\"roles/iam.workloadIdentityUser\" --member=\"principalSet://iam.googleapis.com/projects/\$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')/locations/global/workloadIdentityPools/${WIF_POOL_NAME}/attribute.repository/${GITHUB_REPO}\""
            fi
        fi
    done
}

# Function to create Secret Manager secret for API key
create_secret() {
    echo -e "${YELLOW}🔑 Setting up Secret Manager for API key...${NC}"
    
    # Enable Secret Manager API
    gcloud services enable secretmanager.googleapis.com
    
    # Create secret from api_key.json if it exists
    if [ -f "api_key.json" ]; then
        if gcloud secrets describe gcp-service-account-key &>/dev/null; then
            echo "Updating existing secret..."
            gcloud secrets versions add gcp-service-account-key --data-file=api_key.json
        else
            echo "Creating new secret..."
            gcloud secrets create gcp-service-account-key --data-file=api_key.json
        fi
        
        # Grant access to the service account
        gcloud secrets add-iam-policy-binding gcp-service-account-key \
            --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
            --role="roles/secretmanager.secretAccessor"
        
        echo -e "${GREEN}✓ Secret created and access granted${NC}"
    else
        echo -e "${YELLOW}⚠ api_key.json not found. You'll need to create the secret manually.${NC}"
    fi
}

# Function to output GitHub secrets
output_github_secrets() {
    echo -e "${BLUE}📋 GitHub Repository Secrets to Configure:${NC}"
    echo ""
    echo -e "${YELLOW}Go to your GitHub repository settings > Secrets and variables > Actions${NC}"
    echo -e "${YELLOW}Add the following repository secrets:${NC}"
    echo ""
    
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')
    WIF_PROVIDER_FULL="projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${WIF_POOL_NAME}/providers/${WIF_PROVIDER_NAME}"
    SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"
    
    echo -e "${GREEN}WIF_PROVIDER:${NC} ${WIF_PROVIDER_FULL}"
    echo -e "${GREEN}WIF_SERVICE_ACCOUNT:${NC} ${SERVICE_ACCOUNT_EMAIL}"
    echo ""
    echo -e "${BLUE}Copy these values to your GitHub repository secrets.${NC}"
}

# Function to test deployment
test_deployment() {
    echo -e "${YELLOW}🧪 Testing local Docker build...${NC}"
    
    if docker build -t test-aet-rag .; then
        echo -e "${GREEN}✓ Docker build successful${NC}"
        docker rmi test-aet-rag
    else
        echo -e "${RED}❌ Docker build failed${NC}"
        exit 1
    fi
}

# Main execution
main() {
    echo -e "${BLUE}Starting setup process...${NC}"
    
    check_gcloud_auth
    set_project
    enable_apis
    create_artifact_registry
    create_service_account
    setup_workload_identity
    create_secret
    test_deployment
    
    echo ""
    echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
    echo ""
    
    output_github_secrets
    
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "1. Add the GitHub secrets shown above to your repository"
    echo "2. Commit and push your changes to the main branch"
    echo "3. The GitHub Action will automatically deploy your app to Cloud Run"
    echo ""
    echo -e "${GREEN}Your Cloud Run service will be available at:${NC}"
    echo "https://${SERVICE_NAME}-$(echo $REGION | tr -d '-')-${PROJECT_ID}.a.run.app"
}

# Run the main function
main 