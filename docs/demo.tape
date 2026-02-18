# ollama-scout demo recording
# Record with: vhs docs/demo.tape
# Requires: https://github.com/charmbracelet/vhs

Output docs/demo.gif

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Catppuccin Mocha"
Set Padding 20
Set TypingSpeed 50ms

# --- Scene 1: Full scan (offline) ---
Type "ollama-scout --offline --no-pull-prompt"
Enter
Sleep 3s

# --- Scene 2: Benchmark estimates ---
Type "ollama-scout --offline --benchmark --no-pull-prompt"
Enter
Sleep 4s

# --- Scene 3: Model detail view ---
Type "ollama-scout --offline --model llama3.2 --no-pull-prompt"
Enter
Sleep 3s

# Hold final frame
Sleep 2s
