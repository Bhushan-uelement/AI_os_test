import subprocess
import os
import sys

class TelemetryCollector:
    """Gathers multi-dimensional OS telemetry data robustly."""
    
    @staticmethod
    def run_cmd(cmd, timeout=3, default="N/A"):
        """Run a shell command with a timeout to prevent hanging."""
        try:
            # Execute the bash command and capture its standard output
            # A strict 3-second timeout prevents the script from hanging indefinitely
            # stderr is redirected to DEVNULL to hide any messy bash warnings or errors
            result = subprocess.check_output(cmd, shell=True, timeout=timeout, stderr=subprocess.DEVNULL)
            # Decode the bytes to a UTF-8 string and strip any trailing whitespace
            output = result.decode().strip()
            # If the output is empty, safely fallback to the default string
            return output if output else default
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            # If the command times out, fails, or is missing entirely, return the default
            return default

    def get_hardware_load(self):
        # Runs the "top" command in batch mode for exactly 1 iteration
        # Pipes the output to "head" to only grab the first 5 lines (CPU load, tasks, etc.)
        return self.run_cmd("top -b -n 1 | head -n 5")

    def get_storage_and_memory(self):
        # Combines the "free" command (memory) with the "df" command (root disk usage)
        # Separated by a "---" string for clear formatting in the payload
        return self.run_cmd("free -h && echo '---' && df -h / | tail -1")

    def get_gpu_telemetry(self):
        """Monitor GPU utilization and temperature."""
        # Queries nvidia-smi for specific CSV-formatted metrics (utilization, temp, VRAM)
        gpu_info = self.run_cmd("nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total --format=csv,noheader")
        return gpu_info if gpu_info != "N/A" else "GPU Telemetry Unavailable"

    def get_thermal_data(self):
        """Monitor CPU and system temperatures natively."""
        try:
            # Read the raw temperature values directly from the Linux kernel thermal zones
            temps = subprocess.check_output("cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null", shell=True).decode().strip().split('\n')
            # Read the labels (names) for each corresponding thermal zone
            types = subprocess.check_output("cat /sys/class/thermal/thermal_zone*/type 2>/dev/null", shell=True).decode().strip().split('\n')
            
            results = []
            # Iterate through both lists simultaneously to map labels to temperatures
            for temp, t_type in zip(temps, types):
                if temp.isdigit():
                    # The kernel stores temperatures in millidegrees Celsius, so we divide by 1000
                    celsius = int(temp) / 1000.0
                    # Filter out negative or zero invalid hardware readings
                    if celsius > 0:
                        results.append(f"{t_type}: {celsius:.1f}°C")
            
            if results:
                # Return only the top 5 thermal sensors to keep the prompt length reasonable
                return "\n".join(results[:5])
        except Exception:
            # If we lack permissions or the thermal files don't exist, quietly ignore the failure
            pass
        
        # Fallback message if native thermal reading completely fails
        return "Thermal sensors unavailable"
        
    def get_network_info(self):
        # Displays the statistics for network links (RX/TX bytes and drops)
        return self.run_cmd("ip -s link | head -n 10")

    def get_system_identity(self):
        # Retrieves the static hostname, OS version, and the active local IP addresses
        return self.run_cmd("hostnamectl && echo '---' && ip addr show | grep 'inet ' | head -n 2")

    def get_critical_logs(self):
        # Pulls the 4 most recent kernel/systemd logs labeled with priority 3 (Errors)
        logs = self.run_cmd("journalctl -p 3 -n 4 --no-pager")
        return logs if logs != "N/A" else "No critical errors detected or access restricted."

    def gather_all(self):
        # Notify the user that the background collection process has begun
        print("\033[93m[Gathering MerlinOS Advanced Telemetry at Startup...]\033[0m")
        
        # Assemble a master list of all telemetry components, formatted cleanly with headers
        data = [
            "### HARDWARE LOAD ###\n" + self.get_hardware_load(),
            "### MEMORY & DISK ###\n" + self.get_storage_and_memory(),
            "### GPU METRICS ###\n" + self.get_gpu_telemetry(),
            "### THERMALS ###\n" + self.get_thermal_data(),
            "### SYSTEM IDENTITY ###\n" + self.get_system_identity(),
            "### NETWORK TX/RX ###\n" + self.get_network_info(),
            "### RECENT KERNEL ERRORS ###\n" + self.get_critical_logs()
        ]
        
        # Join the list into a single massive string separated by double newlines
        return "\n\n".join(data)

class MerlinAI:
    """Handles the communication with Llama model."""
    
    def __init__(self, model_path="./Llama-3.2.gguf"):
        # Save the path to the compiled GGUF model binary
        self.model_path = model_path

    def start_conversation(self, telemetry_context):
        # Construct the foundational system instructions for the LLM
        # This tells the AI who it is and how to respond (detailed but not verbose)
        # It then injects the live telemetry data collected earlier directly into the prompt
        system_prompt = (
            f"You are the MerlinOS Intelligence System, an advanced AI monitor for edge computing and robotics. "
            f"Provide detailed and informative answers directly addressing the user's query, but avoid unnecessary verbosity. "
            f"You have the following system telemetry available. Provide context for anomalies if relevant.\n\n"
            f"--- TELEMETRY DATA ---\n{telemetry_context}"
        )
        
        # Prepare the command array to launch the llama.cpp engine
        cmd = [
            "llama-cli", 
            "-m", self.model_path, # Pass the model location
            "-p", system_prompt,   # Set the initial context payload
            "-cnv",                # Enable conversation mode (keeps the model loaded into memory)
            "-n", "512",           # Allow generating up to 512 tokens per response
            "-ngl", "32",          # Offload up to 32 layer calculations to the discrete NVIDIA GPU
            "-c", "4096",          # Establish a 4096 token context window for prolonged memory
            "--temp", "0.2",       # Run with a very low temperature to prevent hallucinatory or creative outputs
            "--log-disable",       # Disable standard verbose internal logging
            "--no-display-prompt", # Prevent the engine from echoing our giant system prompt back to the screen
            "--simple-io"          # Force basic input/output to prevent the CLI from bypassing our terminal redirection
        ]
        
        # Notify the user that the AI is spinning up (since reading multiple GBs to VRAM takes a few seconds)
        print("\033[93m[Loading AI Model... Please wait.]\033[0m")
        try:
            # We spawn a background process with stdout piped directly into our Python script.
            # This allows us to intercept and hide the extremely messy "llama-cli" startup sequence
            p = subprocess.Popen(
                cmd, 
                stdin=subprocess.PIPE, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.DEVNULL, # Mute CUDA initializations and runtime warnings completely
                text=True, 
                bufsize=1
            )
            
            # STEP 1: Swallow all generation text until the engine is fully loaded
            buffer = ""
            while True:
                # Read output one single character at a time from the engine
                char = p.stdout.read(1)
                # If the character is empty, the process likely crashed or exited prematurely
                if not char:
                    break
                buffer += char
                # Wait until the engine prints its internal prompt-timing diagnostic and finishes the `>` prompt
                if "[ Prompt:" in buffer and buffer.endswith("> "):
                    break
                    
            # The AI is now fully loaded and completely silent. Render the green "Ready" alert.
            print("\033[92m[MerlinAI Started. Type your queries below. Press Ctrl+C to exit.]\033[0m\n")
            
            # STEP 2: Start our own clean, invisible chat loop
            while True:
                try:
                    # Print our custom colored user prompt and wait for standard keyboard input
                    user_q = input("\033[96mMerlinOS > \033[0m").strip()
                except EOFError:
                    # If stdin terminates unexpectedly (e.g. from an echo pipe), exit the loop safely
                    break
                
                # Check for standard exit command strings
                if user_q.lower() in ['exit', 'quit', 'q']:
                    # Forcefully kill the background llama-cli process before exiting
                    p.terminate()
                    break
                
                # If the user just pressed "Enter" without typing anything, skip the loop and prompt again
                if not user_q:
                    continue
                    
                # Take the user's string, append a newline, and pipe it directly to llama-cli's internal stdin
                p.stdin.write(user_q + "\n")
                # Flush the pipe immediately to ensure llama-cli receives the text without buffering delays
                p.stdin.flush()
                
                # Retrieve the AI's generated response
                response = ""
                while True:
                    # Read the response one character at a time as it flows out
                    char = p.stdout.read(1)
                    if not char:
                        break
                    
                    response += char
                    
                    # Stop reading once the AI prints its `> ` conversational turning prompt
                    if response.endswith("> "):
                        # Shave off the final "> " from the end of the AI's output string
                        response = response[:-2].strip()
                        
                        # Locate the internal `[ Prompt: XX t/s | Generation: YY t/s ]` debug log added by llama-cli
                        prompt_idx = response.rfind("[ Prompt:")
                        if prompt_idx != -1:
                            # Completely slice the debug log out of the final string for a seamless UI
                            response = response[:prompt_idx].strip()
                        
                        # Break out of the reading block since generation is finished
                        break
                
                # Print the fully sanitized, pure AI response directly to the terminal
                print(f"\n{response}\n")
                
        except KeyboardInterrupt:
            # Handle the user pressing Ctrl+C gracefully without printing huge Python Tracebacks
            print("\n\033[93m[AI Analysis Interrupted by User]\033[0m")
            try:
                # Forcefully kill the background generation process
                p.terminate()
            except:
                pass
        except FileNotFoundError:
            # Triggered if the "llama-cli" application isn't physically installed on the system
            print(f"\033[91mError: 'llama-cli' not found in PATH or '{self.model_path}' is missing.\033[0m")
        except Exception as e:
            # Catch-all for any other fatal communication disruptions
            print(f"\033[91mAn unexpected error occurred during AI analysis: {e}\033[0m")

class MonitorCLI:
    """Main application launcher."""
    
    def __init__(self):
        # Initialize our two primary functional modules
        self.collector = TelemetryCollector()
        self.ai = MerlinAI()

    def display_header(self):
        # Issue a 'clear' command to start with a fresh blank terminal depending on the OS
        os.system('clear' if os.name == 'posix' else 'cls')
        
        # Print out the branding and UI header using magenta terminal color codes
        print("\033[95m" + "="*60)
        print("   MERLIN OS : ADVANCED ARTIFICIAL INTELLIGENCE MONITOR")
        print("            Powered by uElement | Production UI")
        print("="*60 + "\033[0m")

    def run(self):
        # Draw the standard interface header
        self.display_header()
        
        # Sequentially trigger the bash telemetry gathering commands and store the massive payload
        context = self.collector.gather_all()
        
        # Boot up the AI subprocess, feed it the context, and lock the thread into the polling chat loop
        self.ai.start_conversation(context)
        
        # When execution escapes the chat loop (e.g. user typed 'quit'), bid farewell gracefully
        print("\n\033[92mShutting down MerlinOS AI Monitor...\033[0m")

if __name__ == "__main__":
    # Standard Python entry point: instantiate the main GUI class and run it
    app = MonitorCLI()
    app.run()
