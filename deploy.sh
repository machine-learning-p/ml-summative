#!/bin/bash

# Deploy script for WAEMU Banking Risk Assessment API
# This script prepares the project for deployment to Render

echo "🚀 Preparing WAEMU Banking API for Render deployment..."

# Create API directory structure
mkdir -p API/models

# Copy necessary files to API directory
echo "📁 Copying files to API directory..."

# Copy Python files
cp prediction.py API/
cp requirements.txt API/
cp Dockerfile API/
cp .dockerignore API/

# Copy model files (if they exist)
if [ -f "best_model.pkl" ]; then
    cp *.pkl API/models/
    echo "✅ Model files copied"
else
    echo "⚠️  Model files not found. Make sure to run the training notebook first."
fi

# Create a simple test script
cat > API/test_api.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test script for the WAEMU Banking API
"""
import requests
import json

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints"""
    
    print(f"🧪 Testing API at {base_url}")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Response: {response.json()}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check error: {e}")
    
    # Test prediction endpoint
    test_data = {
        "countries_num": 1,
        "year": 2023,
        "rir": 3.5,
        "sfs": 28.5,
        "inf": 2.1,
        "era": 2.8,
        "inl": 10.5,
        "debt": 22.3,
        "size": 12.5,
        "cc": 25.8,
        "ge": 35.2,
        "ps": 45.7,
        "rq": 38.4,
        "rl": 34.6,
        "va": 55.2,
        "countries": "Benin",
        "banks": "Test Bank"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        if response.status_code == 200:
            print("✅ Prediction test passed")
            result = response.json()
            print(f"   Z-Score: {result['predicted_zscore']}")
            print(f"   Risk Level: {result['risk_level']}")
        else:
            print(f"❌ Prediction test failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction test error: {e}")

if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(base_url)
EOF

chmod +x API/test_api.py

# Create local Docker build and run script
cat > API/run_local.sh << 'EOF'
#!/bin/bash

echo "🐳 Building and running WAEMU Banking API locally with Docker..."

# Build the Docker image
docker build -t waemu-banking-api .

if [ $? -eq 0 ]; then
    echo "✅ Docker image built successfully"
    
    # Run the container
    echo "🚀 Starting container on port 8000..."
    docker run -p 8000:8000 --name waemu-api waemu-banking-api
else
    echo "❌ Docker build failed"
    exit 1
fi
EOF

chmod +x API/run_local.sh

# Create Render deployment guide
cat > API/RENDER_DEPLOYMENT.md << 'EOF'
# 🚀 Render Deployment Guide

## Prerequisites
1. GitHub repository with your code
2. Render account (free tier available)
3. Model files (*.pkl) committed to your repository

## Deployment Steps

### Step 1: Prepare Repository
1. Ensure all model files are in the `API/models/` directory
2. Commit and push all changes to GitHub:
   ```bash
   git add .
   git commit -m "Add Docker deployment configuration"
   git push origin main
   ```

### Step 2: Deploy to Render
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure deployment:
   - **Name**: `waemu-banking-api`
   - **Region**: Choose closest to your users
   - **Branch**: `main`
   - **Root Directory**: `API`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Instance Type**: `Free` (or paid for better performance)

### Step 3: Environment Variables (Optional)
Add these environment variables in Render dashboard:
- `PORT`: `8000` (automatically set by Render)
- `PYTHONUNBUFFERED`: `1`

### Step 4: Deploy
1. Click "Create Web Service"
2. Wait for deployment (5-10 minutes)
3. Your API will be available at: `https://your-service-name.onrender.com`

### Step 5: Test Deployment
1. Visit `https://your-service-name.onrender.com/docs` for Swagger UI
2. Test the `/health` endpoint
3. Test the `/predict` endpoint with sample data

## Important Notes

- **Free Tier Limitations**: 
  - Service sleeps after 15 minutes of inactivity
  - 750 hours/month limit
  - Cold start delay (~30 seconds)

- **Model Files**: Ensure your model files are < 500MB total

- **Custom Domain**: Available on paid plans

## Troubleshooting

### Common Issues:
1. **Build Failures**: Check Dockerfile and requirements.txt
2. **Model Loading Errors**: Verify model files are in correct location
3. **Memory Issues**: Upgrade to paid plan if models are large

### Logs:
View deployment logs in Render dashboard for debugging

## Testing Your Deployed API

```bash
# Test health endpoint
curl https://your-service-name.onrender.com/health

# Test prediction endpoint
curl -X POST "https://your-service-name.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "countries_num": 1,
    "year": 2023,
    "rir": 3.5,
    "sfs": 28.5,
    "inf": 2.1,
    "era": 2.8,
    "inl": 10.5,
    "debt": 22.3,
    "size": 12.5,
    "cc": 25.8,
    "ge": 35.2,
    "ps": 45.7,
    "rq": 38.4,
    "rl": 34.6,
    "va": 55.2,
    "countries": "Benin",
    "banks": "Test Bank"
  }'
```
EOF

echo "📦 API directory structure created:"
echo "API/"
echo "├── prediction.py"
echo "├── requirements.txt"
echo "├── Dockerfile"
echo "├── .dockerignore"
echo "├── models/ (for .pkl files)"
echo "├── test_api.py"
echo "├── run_local.sh"
echo "└── RENDER_DEPLOYMENT.md"

echo ""
echo "🔄 Next steps:"
echo "1. Run your Jupyter notebook to generate model files"
echo "2. Copy the .pkl files to API/models/ directory"
echo "3. Test locally: cd API && ./run_local.sh"
echo "4. Follow the deployment guide in API/RENDER_DEPLOYMENT.md"
echo "5. Update your Flutter app with the deployed API URL"

echo ""
echo "✅ Deployment preparation complete!"