# List Custom Seed

**Node Function:** The `List Custom Seed` node generates a list of random seed values based on an input seed. It supports "control after generate" functionality to ensure generated seeds are unique and non-repeating, commonly used for batch generation scenarios requiring different random seeds.

## Inputs

| Parameter Name | Input Selection | Data Type | Default Value | Value Range | Description |
| -------------- | --------------- | --------- | ------------- | ----------- | ----------- |
| `seed` | - | INT | 42 | 0-1125899906842624 | Base seed value for random generation |
| `count` | - | INT | 3 | 1-1000 | Number of seeds to generate |

## Outputs

| Output Name | Data Type | Description |
|-------------|-----------|-------------|
| `seed_list` | INT | Generated list of seed values |
| `count` | INT | Number of generated seeds |