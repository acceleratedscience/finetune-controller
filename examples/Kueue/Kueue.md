# Kueue

### Example `config.json`

The `config.json` file maps the cluster nodes to the appropriate LocalQueue.

Alternatively you can define your own tolerations here and not use Kueue.
```json
{
  // queue here refers to `LocalQueue` definition for Kueue
  // if local_queue not defined will use default_queue for worker. (optional)
  "default_queue": "finetune-queue",
  "workers": [
    {
      // define a worker name. (required)
      "name": "cpu",
      "local_queue": "finetune-queue",
      "defaults": {
        "resources": {
          // default resource requests and limits. (optional)
          "requests": {
            "cpu": 2,
            "memory": "2Gi"
          }
        }
      }
    },
    {
      "name": "gpu-small",
      "local_queue": "finetune-queue",
      "defaults": {
        // define minimum gpu resource requests if any. (optional)
        "accelerators": {
          "nvidia.com/gpu": 1
        }
      },
      // add node tolerations if any. (optional)
      // if using Kueue, define in a `ResourceFlavor` configuration
      "tolerations": [
        {
          "key": "taint-name",
          "value": "cpu",
          "effect": "NoSchedule"
        }
      ]
    }
  ]
}
```
