## Integration tests

These are meant to only be run by our automated tests, or by Kaggle engineers
who are creating or updating tests and want to run locally.

Integration tests should be run by the `integrationtester` Kaggle account. Make
sure you have configured your local `kagglehub` accordingly, via
```
export KAGGLE_USERNAME=integrationtester
export KAGGLE_KEY=...
```
Visit [go/kaggle-integrationtester](http://go/kaggle-integrationtester) to get
the correct `KAGGLE_KEY` value to use in the above command (Actions -> Get
Secret Value).
