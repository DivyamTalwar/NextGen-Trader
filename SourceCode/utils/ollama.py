"""Utilities for working with Ollama models"""

import platform
import subprocess
import requests
import time
from typing import List
import questionary
from colorama import Fore, Style
import os
import re
import webbrowser

# Constants
OLLAMA_SERVER_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_MODELS_ENDPOINT = f"{OLLAMA_SERVER_URL}/api/tags"
OLLAMA_DOWNLOAD_URL = {
    "darwin": "https://ollama.com/download/darwin",
    "windows": "https://ollama.com/download/windows",
    "linux": "https://ollama.com/download/linux",
}


def is_ollama_installed() -> bool:
    """Check if Ollama is installed on the system."""
    system = platform.system().lower()
    command = "where" if system == "windows" else "which"
    
    try:
        # Use shell=True for Windows 'where' command compatibility
        result = subprocess.run([command, "ollama"], capture_output=True, text=True, shell=(system == "windows"))
        return result.returncode == 0
    except (FileNotFoundError, Exception):
        return False


def is_ollama_server_running() -> bool:
    """Check if the Ollama server is running."""
    try:
        response = requests.get(OLLAMA_API_MODELS_ENDPOINT, timeout=2)
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_locally_available_models() -> List[str]:
    """Get a list of models that are already downloaded locally."""
    if not is_ollama_server_running():
        return []

    try:
        response = requests.get(OLLAMA_API_MODELS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except requests.RequestException:
        return []


def start_ollama_server() -> bool:
    """Start the Ollama server if it's not already running."""
    if is_ollama_server_running():
        print(f"{Fore.GREEN}Ollama server is already running.{Style.RESET_ALL}")
        return True

    system = platform.system().lower()
    print(f"{Fore.YELLOW}Starting Ollama server...{Style.RESET_ALL}")

    try:
        # Use Popen for non-blocking start, shell=True for Windows
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=(system == "windows"))
    except Exception as e:
        print(f"{Fore.RED}Error starting Ollama server: {e}{Style.RESET_ALL}")
        return False

    # Wait for the server to become available
    for _ in range(10):  # Try for 10 seconds
        if is_ollama_server_running():
            print(f"{Fore.GREEN}Ollama server started successfully.{Style.RESET_ALL}")
            return True
        time.sleep(1)

    print(f"{Fore.RED}Failed to start Ollama server. Timed out waiting for server to become available.{Style.RESET_ALL}")
    return False


def install_ollama() -> bool:
    """Guides the user through installing Ollama on their system."""
    system = platform.system().lower()
    if system not in OLLAMA_DOWNLOAD_URL:
        print(f"{Fore.RED}Unsupported operating system for automatic installation: {system}{Style.RESET_ALL}")
        print(f"Please visit https://ollama.com/download to install Ollama manually.")
        return False

    # For Linux, attempt direct installation
    if system == "linux":
        print(f"{Fore.YELLOW}Attempting to install Ollama via command line...{Style.RESET_ALL}")
        try:
            install_command = "curl -fsSL https://ollama.com/install.sh | sh"
            process = subprocess.run(install_command, shell=True, check=True, capture_output=True, text=True)
            print(f"{Fore.GREEN}Ollama installed successfully.{Style.RESET_ALL}")
            return True
        except (subprocess.CalledProcessError, Exception) as e:
            print(f"{Fore.RED}Failed to install Ollama. Error: {e}{Style.RESET_ALL}")
            return False

    # For macOS and Windows, guide user to download page
    print(f"{Fore.YELLOW}Ollama for {system.capitalize()} is available as an application download.{Style.RESET_ALL}")
    if questionary.confirm(f"Would you like to open the Ollama download page for {system.capitalize()}?", default=True).ask():
        try:
            webbrowser.open(OLLAMA_DOWNLOAD_URL[system])
            print(f"{Fore.YELLOW}Please download, install, and run the application, then restart this program.{Style.RESET_ALL}")
            if system == "darwin":
                 print(f"{Fore.CYAN}After installation, you may need to open the Ollama app once before continuing.{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Failed to open browser: {e}{Style.RESET_ALL}")
    return False


def download_model(model_name: str) -> bool:
    """Download an Ollama model, showing progress."""
    if not is_ollama_server_running():
        if not start_ollama_server():
            return False

    print(f"{Fore.YELLOW}Downloading model '{model_name}'...{Style.RESET_ALL}")
    print(f"{Fore.CYAN}This may take a while. The download is happening in the background...{Style.RESET_ALL}")

    try:
        process = subprocess.Popen(
            ["ollama", "pull", model_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        # --- Progress Bar Logic ---
        bar_length = 40
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                output = output.strip()
                percentage_match = re.search(r"(\d+(\.\d+)?)%", output)
                if percentage_match:
                    try:
                        percentage = float(percentage_match.group(1))
                        filled_length = int(bar_length * percentage / 100)
                        bar = "█" * filled_length + "░" * (bar_length - filled_length)
                        print(f"\r{Fore.GREEN}{bar}{Style.RESET_ALL} {Fore.YELLOW}{percentage:.1f}%{Style.RESET_ALL}", end="", flush=True)
                    except ValueError:
                        pass # Ignore if conversion fails

        print() # Newline after progress bar finishes
        return_code = process.wait()

        if return_code == 0:
            print(f"{Fore.GREEN}Model '{model_name}' downloaded successfully!{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.RED}Failed to download model '{model_name}'. Please check the model name and your internet connection.{Style.RESET_ALL}")
            return False
    except Exception as e:
        print(f"\n{Fore.RED}Error downloading model '{model_name}': {e}{Style.RESET_ALL}")
        return False


def ensure_ollama_and_model(model_name: str) -> bool:
    """Ensure Ollama is installed, running, and the requested model is available."""
    # 1. Check if Ollama is installed
    if not is_ollama_installed():
        print(f"{Fore.YELLOW}Ollama is not installed on your system.{Style.RESET_ALL}")
        if questionary.confirm("Do you want to install Ollama now?").ask():
            if not install_ollama():
                return False
        else:
            print(f"{Fore.RED}Ollama is required to use local models.{Style.RESET_ALL}")
            return False
    
    # 2. Make sure the server is running
    if not is_ollama_server_running():
        if not start_ollama_server():
            return False
    
    # 3. Check if the model is already downloaded
    available_models = get_locally_available_models()
    if model_name not in available_models:
        print(f"{Fore.YELLOW}Model '{model_name}' is not available locally.{Style.RESET_ALL}")
        
        model_size_info = ""
        if "70b" in model_name:
            model_size_info = " This is a large model and may take a while to download."
        
        if questionary.confirm(f"Do you want to download the '{model_name}' model now?{model_size_info}").ask():
            if not download_model(model_name):
                return False
        else:
            print(f"{Fore.RED}The model '{model_name}' is required to proceed.{Style.RESET_ALL}")
            return False
    
    print(f"{Fore.GREEN}Ollama is ready with model '{model_name}'.{Style.RESET_ALL}")
    return True


def delete_model(model_name: str) -> bool:
    """Delete a locally downloaded Ollama model."""
    if not is_ollama_server_running():
        if not start_ollama_server():
            return False
    
    print(f"{Fore.YELLOW}Deleting model '{model_name}'...{Style.RESET_ALL}")
    
    try:
        process = subprocess.run(["ollama", "rm", model_name], capture_output=True, text=True, check=True)
        print(f"{Fore.GREEN}Model '{model_name}' deleted successfully.{Style.RESET_ALL}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{Fore.RED}Failed to delete model '{model_name}'. Error: {e.stderr.strip()}{Style.RESET_ALL}")
        return False
    except Exception as e:
        print(f"{Fore.RED}Error deleting model '{model_name}': {e}{Style.RESET_ALL}")
        return False


# Command-line usage for utility purposes
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Ollama model manager utility.")
    parser.add_argument("--check-model", help="Check if a model exists and offer to download if needed.")
    parser.add_argument("--delete-model", help="Delete a locally downloaded model.")
    args = parser.parse_args()

    if args.check_model:
        print(f"Ensuring Ollama is installed and model '{args.check_model}' is available...")
        result = ensure_ollama_and_model(args.check_model)
        sys.exit(0 if result else 1)
    elif args.delete_model:
        if questionary.confirm(f"Are you sure you want to delete the model '{args.delete_model}'?").ask():
            result = delete_model(args.delete_model)
            sys.exit(0 if result else 1)
        else:
            print("Deletion cancelled.")
            sys.exit(0)
    else:
        print("No action specified. Use --check-model <model> or --delete-model <model>.")
        sys.exit(1)