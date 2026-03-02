import re
import csv
import sys
import glob

# Find the latest log file if not specified
log_files = glob.glob("../data/OmniNav/log/inference_20260122_120115.log")
if not log_files:
    # Try finding in data dir if moved
    log_files = glob.glob("data/result_iphone_uncropped_enhanced_063a442aee_*/models/chongchongjj/OmniNav/log/inference_*.log")

if not log_files:
    print("Error: Could not find log file.")
    sys.exit(1)

# Use the most recent log file
log_file = sorted(log_files)[-1]
print(f"Processing log file: {log_file}")

output_file = '../data/OmniNav/log/waypoint_data_063a442aee.csv'

with open(log_file, 'r') as f:
    lines = f.readlines()

results = []
current_wp_pred = None  # Will store 5 waypoints [(x,y), ...]

i = 0
while i < len(lines):
    line = lines[i]
    
    # Parse wp_pred raw block (5 lines of coordinates)
    # Example: [DEBUG] wp_pred raw: [[[-0.00149536  0.00735474]
    if '[DEBUG] wp_pred raw:' in line:
        wp_coords = []
        
        # First line
        match = re.search(r'\[\[\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]', line)
        if match:
            wp_coords.append((float(match.group(1)), float(match.group(2))))
        
        # Lines 2-4 (middle waypoints)
        for j in range(1, 4):
            if i + j < len(lines):
                mid_line = lines[i + j].strip()
                # Matches [  x   y ]
                match = re.search(r'\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]', mid_line)
                if match:
                    wp_coords.append((float(match.group(1)), float(match.group(2))))
        
        # Line 5 (last waypoint, ends with ]]])
        if i + 4 < len(lines):
            last_line = lines[i + 4].strip()
            match = re.search(r'\[\s*([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s+([-]?\d+\.?\d*(?:e[-+]?\d+)?)\s*\]\]\]', last_line)
            if match:
                wp_coords.append((float(match.group(1)), float(match.group(2))))
        
        if len(wp_coords) == 5:
            current_wp_pred = wp_coords
        else:
            # Fallback for condensed format (all in one line or fewer lines)
            # This handles cases where numpy might print differently
            pass
            
    # Parse Frame line with headings
    # Example: Frame 0: arrive=0, heading=[-0.92°, -3.39°, ...], sin=[...], cos=[...], wp[0]=...
    if 'Frame ' in line and 'heading=[' in line:
        frame_match = re.search(r'Frame (\d+):', line)
        arrive_match = re.search(r'arrive=(\d+)', line)
        # Note: heading= (singular)
        headings_match = re.search(r'heading=\[([-\d.°,\s]+)\]', line)
        
        if frame_match and arrive_match and headings_match and current_wp_pred:
            frame_idx = int(frame_match.group(1))
            arrive = int(arrive_match.group(1))
            
            # Parse headings list
            headings_str = headings_match.group(1)
            headings = []
            for h in headings_str.split(','):
                h = h.strip().replace('°', '')
                try:
                    headings.append(float(h))
                except:
                    pass
            
            if len(headings) == 5 and len(current_wp_pred) == 5:
                for subframe_idx in range(5):
                    dx, dy = current_wp_pred[subframe_idx]
                    dtheta = headings[subframe_idx]
                    results.append({
                        'frame_idx': frame_idx,
                        'subframe_idx': subframe_idx,
                        'dx': dx,
                        'dy': dy,
                        'dtheta': dtheta,
                        'arrive': arrive
                    })
            
            current_wp_pred = None
    
    i += 1

# Write to CSV
with open(output_file, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['frame_idx', 'subframe_idx', 'dx', 'dy', 'dtheta', 'arrive'])
    writer.writeheader()
    writer.writerows(results)

print(f"Extracted {len(results)} records ({len(results)//5} frames) to {output_file}")