import redis
import json
import os
from pathlib import Path
from aria_node_controller.core.registry_manager import AriaRegistryManager

aria_root = Path("/home/Projects/NH-Mini/sviluppi/ARIA")
redis_host = "192.168.1.120"
redis_port = 6379

r = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
manager = AriaRegistryManager(aria_root, r)
manager.publish()

print("Registry published successfully.")
registry = r.get("aria:registry:master")
reg_data = json.loads(registry)
print(f"Node: {reg_data['node_id']}")
print(f"Voices found: {len(reg_data['assets']['voices'])}")

# Check one voice
if reg_data['assets']['voices']:
    first_voice = next(iter(reg_data['assets']['voices']))
    print(f"Sample path for {first_voice}: {reg_data['assets']['voices'][first_voice].get('sample_path')}")
