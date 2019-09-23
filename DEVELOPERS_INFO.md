### Info for HOST developers
The test will pass only if the `host.picker` module will have the following lines uncommented (active)

```
        # MB: smooth HOS_CF (next line)
        hos_arr = HS.transform_f4(hos_arr, N, window_type='hanning')
```

In further release should be added some switch in the configuration file to decide which steps between
`f1-f2-f3-f4` should be done 
