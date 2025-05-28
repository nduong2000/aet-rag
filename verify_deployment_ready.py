#!/usr/bin/env python3
"""
Deployment Readiness Verification Script
Checks that all components are ready for Cloud Run deployment via GitHub Actions
"""

import os
import json
import subprocess
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status"""
    if os.path.exists(filepath):
        print(f"   ✅ {description}: {filepath}")
        return True
    else:
        print(f"   ❌ {description}: {filepath} - NOT FOUND")
        return False

def check_directory_exists(dirpath, description):
    """Check if a directory exists and report status"""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        file_count = len(os.listdir(dirpath))
        print(f"   ✅ {description}: {dirpath} ({file_count} files)")
        return True
    else:
        print(f"   ❌ {description}: {dirpath} - NOT FOUND")
        return False

def check_gitignore():
    """Check .gitignore configuration"""
    print("\n🔒 Checking .gitignore Security Configuration:")
    
    if not os.path.exists('.gitignore'):
        print("   ❌ .gitignore file not found")
        return False
    
    with open('.gitignore', 'r') as f:
        gitignore_content = f.read()
    
    required_entries = [
        'api_key.json',
        'service-account-key.json',
        '*.json',
        'credentials/',
        'secrets/'
    ]
    
    all_present = True
    for entry in required_entries:
        if entry in gitignore_content:
            print(f"   ✅ {entry} is properly excluded")
        else:
            print(f"   ⚠️  {entry} not found in .gitignore")
            all_present = False
    
    return all_present

def check_dockerfile():
    """Check Dockerfile configuration"""
    print("\n🐳 Checking Dockerfile Configuration:")
    
    if not os.path.exists('Dockerfile'):
        print("   ❌ Dockerfile not found")
        return False
    
    with open('Dockerfile', 'r') as f:
        dockerfile_content = f.read()
    
    required_copies = [
        'COPY ./templates ./templates',
        'COPY ./chroma_db_data ./chroma_db_data',
        'COPY ./documents ./documents',
        'COPY main.py .',
        'COPY field_definitions.json .'
    ]
    
    all_present = True
    for copy_cmd in required_copies:
        if copy_cmd in dockerfile_content:
            print(f"   ✅ {copy_cmd}")
        else:
            print(f"   ❌ Missing: {copy_cmd}")
            all_present = False
    
    # Check that api_key.json is NOT copied
    if 'api_key.json' not in dockerfile_content:
        print("   ✅ api_key.json correctly excluded from Docker image")
    else:
        print("   ❌ api_key.json should NOT be copied to Docker image")
        all_present = False
    
    return all_present

def check_github_workflow():
    """Check GitHub Actions workflow"""
    print("\n🔄 Checking GitHub Actions Workflow:")
    
    workflow_path = '.github/workflows/google-cloudrun-docker.yml'
    if not os.path.exists(workflow_path):
        print("   ❌ GitHub workflow file not found")
        return False
    
    with open(workflow_path, 'r') as f:
        workflow_content = f.read()
    
    checks = [
        ('us-central1', 'Region set to us-central1'),
        ('aethrag2', 'Project ID set to aethrag2'),
        ('GOOGLE_CLOUD_PROJECT', 'Environment variable configured'),
        ('GCP_LOCATION', 'Location variable configured'),
        ('workload_identity_provider', 'Workload Identity configured')
    ]
    
    all_present = True
    for check_text, description in checks:
        if check_text in workflow_content:
            print(f"   ✅ {description}")
        else:
            print(f"   ❌ {description} - not found")
            all_present = False
    
    return all_present

def check_authentication_setup():
    """Check authentication configuration in main.py"""
    print("\n🔐 Checking Authentication Setup:")
    
    if not os.path.exists('main.py'):
        print("   ❌ main.py not found")
        return False
    
    with open('main.py', 'r') as f:
        main_content = f.read()
    
    auth_checks = [
        ('K_SERVICE', 'Cloud Run detection'),
        ('Workload Identity Federation', 'Cloud Run auth method'),
        ('us-central1', 'Region configuration'),
        ('gemini-1.5-flash-001', 'Compatible model'),
        ('_GLOBAL_CREDENTIALS', 'Global credential management'),
        ('fallback logic', 'Model fallback handling')
    ]
    
    all_present = True
    for check_text, description in auth_checks:
        if check_text in main_content:
            print(f"   ✅ {description}")
        else:
            print(f"   ⚠️  {description} - check manually")
    
    return True

def check_local_files():
    """Check that required local files exist but are not tracked"""
    print("\n📁 Checking Local Development Files:")
    
    # Check that api_key.json exists locally
    if os.path.exists('api_key.json'):
        print("   ✅ api_key.json exists locally for development")
        
        # Check if it's tracked by git
        try:
            result = subprocess.run(['git', 'ls-files', 'api_key.json'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("   ❌ api_key.json is tracked by Git - should be untracked!")
                return False
            else:
                print("   ✅ api_key.json is properly untracked by Git")
        except Exception as e:
            print(f"   ⚠️  Could not check Git status: {e}")
    else:
        print("   ⚠️  api_key.json not found locally - needed for development")
    
    return True

def main():
    print("🚀 AET-RAG Cloud Run Deployment Readiness Check")
    print("=" * 60)
    
    all_checks_passed = True
    
    # Check essential files
    print("\n📋 Checking Essential Files:")
    essential_files = [
        ('main.py', 'Main application file'),
        ('requirements-main.txt', 'Python dependencies'),
        ('Dockerfile', 'Docker configuration'),
        ('templates/chat.html', 'Chat interface template'),
        ('field_definitions.json', 'Field definitions data')
    ]
    
    for filepath, description in essential_files:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    
    # Check essential directories
    print("\n📂 Checking Essential Directories:")
    essential_dirs = [
        ('templates', 'Flask templates'),
        ('chroma_db_data', 'Vector database'),
        ('documents', 'Source documents'),
        ('.github/workflows', 'GitHub Actions')
    ]
    
    for dirpath, description in essential_dirs:
        if not check_directory_exists(dirpath, description):
            all_checks_passed = False
    
    # Run specific checks
    if not check_gitignore():
        all_checks_passed = False
    
    if not check_dockerfile():
        all_checks_passed = False
    
    if not check_github_workflow():
        all_checks_passed = False
    
    if not check_authentication_setup():
        all_checks_passed = False
    
    if not check_local_files():
        all_checks_passed = False
    
    # Final summary
    print("\n" + "=" * 60)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Ready for Cloud Run deployment via GitHub Actions")
        print("\n📝 Next Steps:")
        print("   1. Commit and push changes to main branch")
        print("   2. GitHub Actions will automatically deploy to Cloud Run")
        print("   3. Monitor deployment in GitHub Actions tab")
        print("   4. Check Cloud Run logs if needed")
    else:
        print("❌ SOME CHECKS FAILED!")
        print("⚠️  Please fix the issues above before deploying")
    
    print("\n🔗 Useful Commands:")
    print("   - Check deployment: gcloud run services list")
    print("   - View logs: gcloud logging read 'resource.type=cloud_run_revision'")
    print("   - Test locally: python main.py")
    
    return all_checks_passed

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 