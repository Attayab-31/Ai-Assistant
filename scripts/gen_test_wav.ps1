Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile("$PSScriptRoot\..\test_utterance.wav")
$s.Speak("Two adults and no children")
$s.Dispose()
