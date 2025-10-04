"""
generateTrainingData.py

Main script for generating training conversations using multiple API providers.
Supports async generation with tool execution.

Usage:
    python data/generateTrainingData.py [--target N] [--providers mistral,together]
"""

import asyncio
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from secrets_manager import SecretsManager
from api_clients import create_client
from tool_executor import ToolExecutor
from conversation_generator import ConversationGenerator


# =============================
# Configuration
# =============================
OUTPUT_FILE = "data/trainingData/generatedTrainingData.json"
DEFAULT_TARGET = 5000


# =============================
# Progress Tracking
# =============================
class ProgressTracker:
    """Track and display generation progress"""
    
    def __init__(self, target: int, existing: int = 0):
        self.target = target
        self.current = existing
        self.start_time = datetime.now()
        self.provider_counts = {}
    
    def increment(self, provider: str):
        """Increment count for a provider"""
        self.current += 1
        self.provider_counts[provider] = self.provider_counts.get(provider, 0) + 1
    
    def print_status(self):
        """Print current progress"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        remaining = (self.target - self.current) / rate if rate > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"📊 Progress: {self.current}/{self.target} ({self.current/self.target*100:.1f}%)")
        print(f"⚡ Rate: {rate*3600:.0f} conversations/hour")
        print(f"⏱️  Estimated time remaining: {remaining/3600:.1f} hours")
        
        if self.provider_counts:
            print(f"\n📈 Provider breakdown:")
            for provider, count in sorted(self.provider_counts.items()):
                print(f"   {provider}: {count}")
        
        print(f"{'='*60}\n")
    
    def is_complete(self) -> bool:
        """Check if target reached"""
        return self.current >= self.target


# =============================
# Multi-Provider Manager
# =============================
class MultiProviderManager:
    """Manages multiple API providers with round-robin distribution"""
    
    def __init__(self, secrets_manager: SecretsManager, tool_executor: ToolExecutor):
        self.secrets_manager = secrets_manager
        self.tool_executor = tool_executor
        self.generators = []
        self.current_index = 0
        
        self._initialize_generators()
    
    def _initialize_generators(self):
        """Initialize conversation generators for enabled providers"""
        enabled_providers = self.secrets_manager.get_enabled_providers()
        
        print("🔧 Initializing API clients...")
        
        for provider in enabled_providers:
            api_key = self.secrets_manager.get_key(provider)
            
            if not api_key:
                print(f"⚠️  Skipping {provider}: No API key configured")
                continue
            
            try:
                client = create_client(provider, api_key)
                generator = ConversationGenerator(client, self.tool_executor)
                self.generators.append((provider, generator))
                print(f"✅ {provider.capitalize()} client ready")
            except Exception as e:
                print(f"❌ Failed to initialize {provider}: {e}")
        
        if not self.generators:
            raise RuntimeError("No API clients initialized! Check your secrets.json")
        
        print(f"\n🚀 Ready with {len(self.generators)} provider(s)\n")
    
    def get_next_generator(self) -> tuple:
        """Get next generator in round-robin fashion"""
        if not self.generators:
            raise RuntimeError("No generators available")
        
        provider, generator = self.generators[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.generators)
        
        return provider, generator
    
    def get_provider_count(self) -> int:
        """Get number of active providers"""
        return len(self.generators)


# =============================
# File Operations
# =============================
def count_existing_conversations(filepath: str) -> int:
    """Count conversations in existing file"""
    if not os.path.exists(filepath):
        return 0
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            return content.count('"conversations"')
    except:
        return 0


def load_last_conversation(filepath: str) -> tuple:
    """Load last conversation for context"""
    if not os.path.exists(filepath):
        return "None.", "None."
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            # Find last complete conversation
            # This is a simple approach - could be improved
            if '"conversations"' in content:
                return "None.", "None."  # For now, just start fresh
    except:
        pass
    
    return "None.", "None."


def save_conversation(filepath: str, conversation: Dict):
    """Append conversation to file"""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'a') as f:
        json.dump(conversation['conversations'], f)
        f.write(",\n")


# =============================
# Main Generation Loop
# =============================
async def generate_conversations(
    manager: MultiProviderManager,
    target: int,
    output_file: str
):
    """Main async generation loop"""
    
    # Setup
    existing_count = count_existing_conversations(output_file)
    prev_question, prev_answer = load_last_conversation(output_file)
    
    tracker = ProgressTracker(target, existing_count)
    
    print(f"📊 Starting generation:")
    print(f"   Existing: {existing_count}")
    print(f"   Target: {target}")
    print(f"   To generate: {target - existing_count}")
    print(f"   Providers: {manager.get_provider_count()}")
    print()
    
    # Generation loop
    try:
        while not tracker.is_complete():
            provider, generator = manager.get_next_generator()
            
            # Generate conversation
            conversation = await generator.generate_conversation(
                previous_question=prev_question,
                previous_answer=prev_answer
            )
            
            if conversation:
                # Save to file
                save_conversation(output_file, conversation)
                
                # Update context
                prev_question = conversation['conversations'][1]['content']
                prev_answer = conversation['conversations'][2]['content']
                
                # Track progress
                tracker.increment(provider)
                
                # Print status every 10 conversations
                if tracker.current % 10 == 0:
                    tracker.print_status()
                    
                    # Print sample
                    if tracker.current % 50 == 0:
                        print(f"📝 Sample question: {prev_question[:80]}...")
                        print()
            else:
                print(f"⚠️  Failed to generate conversation with {provider}")
    
    except KeyboardInterrupt:
        print("\n⏸️  Generation paused by user")
    
    finally:
        # Final status
        tracker.print_status()
        
        # Tool stats
        stats = manager.tool_executor.get_stats()
        print(f"🔧 Tool usage statistics:")
        print(f"   Memories: {stats['memories']}")
        print(f"   Diary pages: {stats['diary_pages']}")
        print(f"\n✅ Generation complete!")
        print(f"💾 Output saved to: {output_file}")


# =============================
# CLI Interface
# =============================
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate training conversations for APEP"
    )
    
    parser.add_argument(
        '--target',
        type=int,
        default=DEFAULT_TARGET,
        help=f'Target number of conversations (default: {DEFAULT_TARGET})'
    )
    
    parser.add_argument(
        '--providers',
        type=str,
        help='Comma-separated list of providers to use (overrides secrets.json)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=OUTPUT_FILE,
        help=f'Output file path (default: {OUTPUT_FILE})'
    )
    
    parser.add_argument(
        '--reset-tools',
        action='store_true',
        help='Reset tool database before starting'
    )
    
    return parser.parse_args()


async def main():
    """Main entry point"""
    args = parse_args()
    
    print("🌸 APEP Training Data Generator")
    print("=" * 60)
    
    # Load secrets
    secrets_manager = SecretsManager()
    
    # Override providers if specified
    if args.providers:
        providers = [p.strip() for p in args.providers.split(',')]
        print(f"🔧 Using providers: {', '.join(providers)}")
        # Update enabled providers in secrets
        secrets_manager.secrets['enabled_providers'] = providers
    
    # Initialize tool executor
    tool_executor = ToolExecutor()
    
    if args.reset_tools:
        print("⚠️  Resetting tool database...")
        tool_executor.reset_database()
    
    # Initialize manager
    try:
        manager = MultiProviderManager(secrets_manager, tool_executor)
    except RuntimeError as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Make sure you have:")
        print("   1. Added API keys to data/secrets.json")
        print("   2. Enabled at least one provider in 'enabled_providers'")
        return
    
    # Generate conversations
    await generate_conversations(
        manager=manager,
        target=args.target,
        output_file=args.output
    )
    
    # Cleanup
    tool_executor.close()


if __name__ == "__main__":
    asyncio.run(main())