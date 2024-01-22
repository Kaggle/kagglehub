# Changelog

## Next (Unreleased)

*

## v0.1.6 (January 22nd, 2023)

* Fix permission issue in `model_upload` and add integration tests ([#69](https://github.com/Kaggle/kagglehub/pull/69))
* Make specifying a license optional in `model_upload` ([#62](https://github.com/Kaggle/kagglehub/pull/62))
* Improve logging ([#68](https://github.com/Kaggle/kagglehub/pull/68), [#71](https://github.com/Kaggle/kagglehub/pull/71))
* Add resumable upload ([#55](https://github.com/Kaggle/kagglehub/pull/55))

## v0.1.5 (January 8th, 2023)

* Prevent log message from being printed twice in some environment ([#57](https://github.com/Kaggle/kagglehub/pull/57))
* Add Colab model resolver ([#53](https://github.com/Kaggle/kagglehub/pull/53))
* Add `kagglehub.model_upload(...)` ([#43](https://github.com/Kaggle/kagglehub/pull/43), [#51](https://github.com/Kaggle/kagglehub/pull/51), [#52](https://github.com/Kaggle/kagglehub/pull/52))
* Add `kagglehub` user agent to Kaggle API V1 calls ([#50](https://github.com/Kaggle/kagglehub/pull/50))
* Add `force_download` option to `kagglehub.model_download()` ([#44](https://github.com/Kaggle/kagglehub/pull/44))

## v0.1.4 (Dec 11th, 2023)

* Improve error messages for `KaggleCacheResolver` ([#40](https://github.com/Kaggle/kagglehub/pull/40))

## v0.1.3 (Dec 5th, 2023)

* Improve error messages for Kaggle API calls ([#38](https://github.com/Kaggle/kagglehub/pull/38))
* Perform integrity check after file download ([#37](https://github.com/Kaggle/kagglehub/pull/37))

## v0.1.2 (Nov 30th, 2023)

* Fixed notebook environment detection logic ([#36](https://github.com/Kaggle/kagglehub/pull/36))

## v0.1.1 (Nov 30th, 2023)

* Fixed login credential validation ([#33](https://github.com/Kaggle/kagglehub/pull/33), [#34](https://github.com/Kaggle/kagglehub/pull/34))

## v0.1.0 (Nov 29th, 2023)

* Attach model in Kaggle notebook environment with internet disabled ([#27](https://github.com/Kaggle/kagglehub/pull/27))
* Login via IPyWidgets in notebook ([#28](https://github.com/Kaggle/kagglehub/pull/28))
* Login via prompt in terminal ([#23](https://github.com/Kaggle/kagglehub/pull/23))
* Attach model in Kaggle Notebook environment ([#19](https://github.com/Kaggle/kagglehub/pull/19))
* Support resumable download ([#17](https://github.com/Kaggle/kagglehub/pull/17))
* Support unversioned model handles ([#16](https://github.com/Kaggle/kagglehub/pull/16))

## v0.0.1a1 (Oct 26th, 2023)

* Login via environment variable or credentials file ([#9](https://github.com/Kaggle/kagglehub/pull/9))
* Download public models over HTTP and store in local cache ([#8](https://github.com/Kaggle/kagglehub/pull/8), [#12](https://github.com/Kaggle/kagglehub/pull/12))

## v0.0.1a0 (Oct 5th, 2023)

* Skeleton for the kagglehub library ([#1](https://github.com/Kaggle/kagglehub/pull/1))