from modules.distraction_counter import DistractionCounter
import time

tracker = DistractionCounter()

# Simulate focus changes
sequence = ['High','High','High','Low','Low',
            'High','Medium','Low','High','Low']

for label in sequence:
    result = tracker.update(label)
    if result:
        print(f"Distraction detected! Total: {tracker.get_count()}")
    time.sleep(6)  # wait past cooldown

print("\nSummary:", tracker.get_summary(60))