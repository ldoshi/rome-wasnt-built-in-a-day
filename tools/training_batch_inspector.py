import collections

from bridger.logging_utils import object_log_readers

log_file = "../object_logging/20240222_222719/training_history_td_error"


target_batch_idx = 3213

def get_entries(target_batch_idx: int) -> list:
    entries = []
    for entry in object_log_readers.read_object_log(log_file):
        if entry.batch_idx == target_batch_idx:
            entries.append(entry)
        
    return entries


entries = get_entries(target_batch_idx)

repeated_entries = collections.defaultdict(list)
for entry in entries:
    repeated_entries[(entry.state_id, entry.action)].append(entry)

for key, repeated in repeated_entries.items():
    if len(repeated) == 1:
        continue

    td_error = repeated[0].td_error
    for entry in repeated:
        if entry.td_error != td_error:
            print(f"{key}: {td_error} vs {entry.td_error}")
    

    
