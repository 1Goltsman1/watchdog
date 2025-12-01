#!/bin/bash
# Script to push to GitHub after creating private repository

echo "=== Bike Monitor - GitHub Push Script ==="
echo ""
echo "Before running this script, please:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: BikeMonitor (or your preferred name)"
echo "3. Description: AI-powered bike theft prevention system with Hailo-8"
echo "4. Select: PRIVATE"
echo "5. Do NOT initialize with README (we already have one)"
echo "6. Click 'Create repository'"
echo ""
read -p "Enter your GitHub username: " username
read -p "Enter the repository name (default: BikeMonitor): " reponame
reponame=${reponame:-BikeMonitor}

echo ""
echo "Setting up remote..."
git remote add origin "https://github.com/$username/$reponame.git"

echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✓ Done! Your repository is now at:"
echo "  https://github.com/$username/$reponame"
echo ""
echo "Note: You may be prompted for your GitHub credentials."
echo "If you have 2FA enabled, use a Personal Access Token instead of your password."
echo "Generate one at: https://github.com/settings/tokens"
