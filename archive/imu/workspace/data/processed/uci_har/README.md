# Processed UCI HAR Train Split

`X_train_raw_float32.bin` stores the training split as a contiguous float32 array.

Shape:

```text
7352 x 128 x 9
```

Dimension meaning:

- 7352: training window samples
- 128: time readings inside each 2.56 second window
- 9: IMU channels listed in `channel_names.txt`

Later, if NumPy is available, it can be loaded with:

```python
import numpy as np
X = np.fromfile('X_train_raw_float32.bin', dtype='<f4').reshape(7352, 128, 9)
```
