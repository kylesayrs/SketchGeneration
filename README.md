TODO:
1. pen_state is not being learned as robustly as I'd hope. Increasing its weight in the loss function helps, but the value still doesn't vary much with respect to inputs.

2. differring outputs at visualization time. This may be because we're autoregressing, but the outputs are suspiciously high when compared to the outputs produced at training time.
