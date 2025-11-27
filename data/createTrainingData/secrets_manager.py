"""
secrets_manager.py

Centralized secrets and API key management.
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional
CREATE_TRAINING_DATA_PATH="data/createTrainingData"

class SecretsManager:
    """Manages API keys and secrets for training data generation"""
    
    def __init__(self, secrets_file: str = "data/createTrainingData/secrets.json"):
        self.secrets_file = Path(secrets_file)
        self.secrets = {}
        self._load_or_create()
    
    def _load_or_create(self):
        """Load existing secrets or create template"""
        if not self.secrets_file.exists():
            self._create_template()
            print(f"❌ Secrets file not found!")
            print(f"📝 Created template at: {self.secrets_file}")
            print(f"\n⚠️  Please add your API keys to: {self.secrets_file}")
            print("   Then restart the script.")
            return
        
        try:
            with open(self.secrets_file, 'r') as f:
                self.secrets = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error parsing secrets file: {e}")
            print(f"\n💡 Your secrets.json has invalid JSON syntax.")
            print(f"   Common issues:")
            print(f"   - Trailing commas (not allowed in JSON)")
            print(f"   - Missing quotes around strings")
            print(f"   - Unescaped special characters")
            print(f"\n🔧 Fix the file or delete it to regenerate: {self.secrets_file}")
            raise
        
        # Validate secrets
        self._validate_secrets()
    
    def _create_template(self):
        """Create secrets template file"""
        self.secrets_file.parent.mkdir(parents=True, exist_ok=True)
        
        template = {
            "mistral_api_key": "YOUR_MISTRAL_KEY_HERE",
            "together_api_key": "YOUR_TOGETHER_KEY_HERE",
            "huggingface_api_key": "YOUR_HF_KEY_HERE",
            "notes": {
                "mistral": "Get key from: https://console.mistral.ai/",
                "together": "Get key from: https://api.together.xyz/",
                "huggingface": "Get key from: https://huggingface.co/settings/tokens"
            },
            "enabled_providers": [
                "mistral"
            ]
        }
        
        with open(self.secrets_file, 'w') as f:
            json.dump(template, f, indent=2)
    
    def _validate_secrets(self):
        """Check if required API keys are set"""
        placeholder_values = ["YOUR_MISTRAL_KEY_HERE", "YOUR_TOGETHER_KEY_HERE", "YOUR_HF_KEY_HERE"]
        
        for key, value in self.secrets.items():
            if key.endswith("_api_key") and value in placeholder_values:
                print(f"⚠️  Warning: {key} not configured")
    
    def get_key(self, provider: str) -> Optional[str]:
        """Get API key for specific provider"""
        key_name = f"{provider}_api_key"
        key = self.secrets.get(key_name)
        
        if not key or key.startswith("YOUR_"):
            return None
        
        return key
    
    def has_key(self, provider: str) -> bool:
        """Check if provider key is configured"""
        return self.get_key(provider) is not None
    
    def get_enabled_providers(self) -> list:
        """Get list of enabled providers"""
        return self.secrets.get("enabled_providers", ["mistral"])
    
    def is_provider_enabled(self, provider: str) -> bool:
        """Check if provider is enabled"""
        return provider in self.get_enabled_providers()
    
    def add_provider(self, provider: str):
        """Enable a provider"""
        enabled = self.get_enabled_providers()
        if provider not in enabled:
            enabled.append(provider)
            self.secrets["enabled_providers"] = enabled
            self._save()
    
    def remove_provider(self, provider: str):
        """Disable a provider"""
        enabled = self.get_enabled_providers()
        if provider in enabled:
            enabled.remove(provider)
            self.secrets["enabled_providers"] = enabled
            self._save()
    
    def _save(self):
        """Save secrets back to file"""
        with open(self.secrets_file, 'w') as f:
            json.dump(self.secrets, f, indent=2)
