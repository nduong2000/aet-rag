#!/bin/bash

# Quick fix script to complete the setup after service account creation
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration (same as main script)
PROJECT_ID="aethrag2"
REGION="us-east1"
SERVICE_NAME="aet-rag-service"
REPOSITORY_NAME="aet-rag-repo"
WIF_POOL_NAME="github-pool"
WIF_PROVIDER_NAME="github-provider"
SERVICE_ACCOUNT_NAME="github-actions-sa"
GITHUB_REPO="nduong2000/aet-rag"

echo -e "${BLUE}🔧 Fixing IAM role assignments...${NC}"

# Grant necessary roles with retry logic
echo "Granting IAM roles to service account..."
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
            echo -e "${GREEN}✓ Successfully granted $role${NC}"
            break
        else
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retries ]; then
                echo "Retrying in 5 seconds... (attempt $retry_count/$max_retries)"
                sleep 5
            else
                echo -e "${RED}❌ Failed to grant $role after $max_retries attempts${NC}"
            fi
        fi
    done
done

echo -e "${GREEN}✓ IAM roles assignment completed${NC}"

# Continue with Workload Identity Federation setup
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

# Allow GitHub Actions to impersonate the service account
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
            echo "Retrying in 5 seconds... (attempt $retry_count/$max_retries)"
            sleep 5
        else
            echo -e "${RED}❌ Failed to configure Workload Identity binding${NC}"
        fi
    fi
done

# Set up Secret Manager
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

# Output GitHub secrets
echo ""
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

echo ""
echo -e "${GREEN}🎉 Setup completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Add the GitHub secrets shown above to your repository"
echo "2. Commit and push your changes to the main branch"
echo "3. The GitHub Action will automatically deploy your app to Cloud Run"
echo ""
echo -e "${GREEN}Your Cloud Run service will be available at:${NC}"
echo "https://${SERVICE_NAME}-$(echo $REGION | tr -d '-')-${PROJECT_ID}.a.run.app" 