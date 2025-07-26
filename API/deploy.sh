#!/usr/bin/env python3
"""
Quick deployment setup for WAEMU Banking Risk Assessment API
Run this script to prepare your files for Render deployment
"""

import os
import shutil

def main():
    print("🚀 WAEMU Banking API - Quick Deployment Setup")
    print("=" * 50)
    
    # Check if model files exist
    required_files = [
        'best_model.pkl',
        'scaler.pkl', 
        'le_countries.pkl',
        'le_banks.pkl',
        'feature_names.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Please run your training notebook first to generate these files.")
        return
    
    print("✅ All model files found!")
    
    # Create deployment files
    print("\n📝 Creating deployment files...")
    
    # Create requirements.txt
    requirements = """fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
scikit-learn==1.3.2
numpy==1.25.2
pandas==2.1.4
joblib==1.3.2"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("   ✅ requirements.txt created")
    
    # Create Dockerfile
    dockerfile = """FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "prediction.py"]"""
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    print("   ✅ Dockerfile created")
    
    # Check if prediction.py exists and is the simplified version
    if not os.path.exists('prediction.py'):
        print("   ⚠️  prediction.py not found - please copy the simplified version")
    else:
        print("   ✅ prediction.py found")
    
    print("\n📋 Deployment Checklist:")
    print("1. ✅ Model files present")
    print("2. ✅ requirements.txt created") 
    print("3. ✅ Dockerfile created")
    
    if os.path.exists('prediction.py'):
        print("4. ✅ prediction.py present")
    else:
        print("4. ❌ prediction.py missing")
    
    print("\n🚀 Next Steps:")
    print("1. Copy the simplified prediction.py code")
    print("2. Commit all files to GitHub:")
    print("   git add .")
    print("   git commit -m 'Add API deployment files'")
    print("   git push origin main")
    print("3. Deploy to Render:")
    print("   - Go to render.com")
    print("   - New Web Service")
    print("   - Connect your GitHub repo")
    print("   - Environment: Docker")
    print("   - Deploy!")
    print("\n✨ Your API will be live at: https://your-service-name.onrender.com")

if __name__ == "__main__":
    main()