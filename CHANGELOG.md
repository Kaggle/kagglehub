# Changelog

## v0.3.5 (December 10th, 2024)

* Include server-side errors if present ([#183](https://github.com/Kaggle/kagglehub/pull/183))
* Made file logging opt-out by default ([#190](https://github.com/Kaggle/kagglehub/pull/190))
* Improved version upgrade warning ([#188](https://github.com/Kaggle/kagglehub/pull/188))
* Add `kagglehub.notebook_output_download(...)`, only supports HTTP resolver for now. ([#184](https://github.com/Kaggle/kagglehub/pull/184))

## v0.3.4 (November 8th, 2024)

* Add a naive Colab dataset cache resolver ([#177](https://github.com/Kaggle/kagglehub/pull/177))
* Add `keras_hub` as user-agent ([#176](https://github.com/Kaggle/kagglehub/pull/176))

## v0.3.3 (October 17th, 2024)

* Hide API key in terminal when prompting in `kaggle.login(...)` ([#173](https://github.com/Kaggle/kagglehub/pull/173))
* Reduce logging spam when using `force_download` in Kaggle notebook environment ([#172](https://github.com/Kaggle/kagglehub/pull/172))

## v0.3.2 (October 15th, 2024)

* Fallback to next auth mechanism if auth environment variables are set but empty ([#168](https://github.com/Kaggle/kagglehub/pull/168))
* `kaggle.whoami()` returns proper value in Kaggle Notebook environment ([#166](https://github.com/Kaggle/kagglehub/pull/166))
* `kaggle.whoami()` returns proper value when using Colab secrets ([#165](https://github.com/Kaggle/kagglehub/pull/165))

## v0.3.1 (October 1st, 2024)

* Attach competition in Kaggle notebook environment ([#161](https://github.com/Kaggle/kagglehub/pull/161))
* Download competition using `kagglehub.competition_download(...)` ([#158](https://github.com/Kaggle/kagglehub/pull/158))
* Enable authorization via Colab secrets ([#157](https://github.com/Kaggle/kagglehub/pull/157))

## v0.3.0 (September 4th, 2024)

* Integrate Keras metrics ([#152](https://github.com/Kaggle/kagglehub/pull/152))
* Enhance logging messages ([#151](https://github.com/Kaggle/kagglehub/pull/151))
* Upgrade Python version support to 3.9 and above ([#154](https://github.com/Kaggle/kagglehub/pull/154))

## v0.2.9 (July 31st, 2024)

* Improve upload by ignoring patterns ([#147](https://github.com/Kaggle/kagglehub/pull/147))
* Upload with empty files ([#145](https://github.com/Kaggle/kagglehub/pull/145))
* Attach dataset in Kaggle notebook environment ([#134](https://github.com/Kaggle/kagglehub/pull/134))
* Upload datasets using `kagglehub.dataset_upload(...)` ([#149](https://github.com/Kaggle/kagglehub/pull/149))
* Download datasets using `kagglehub.dataset_download(...)` ([#131](https://github.com/Kaggle/kagglehub/pull/131))

## v0.2.8 (July 16th, 2024)

* Added support for implicit token auth in kaggle notebooks ([#141](https://github.com/Kaggle/kagglehub/pull/141))
* Improve logging for cached models in Kaggle & Colab environments ([#142](https://github.com/Kaggle/kagglehub/pull/142))

## v0.2.7 (July 1st, 2024)
* Fix model downloading bug

## v0.2.6 (June 10th, 2024)

* Add support for `.zip` archive ([#135](https://github.com/Kaggle/kagglehub/pull/135))
* Reduce spammy logging ([#124](https://github.com/Kaggle/kagglehub/pull/124))
* Download model files in parallel ([#122](https://github.com/Kaggle/kagglehub/pull/122))

## v0.2.5 (May 2nd, 2024)

* Add missing packaging dependencies ([#115](https://github.com/Kaggle/kagglehub/pull/115))
* Create whoami method ([#114](https://github.com/Kaggle/kagglehub/pull/114))
* Migrate all tests to use Flask ([#111](https://github.com/Kaggle/kagglehub/pull/111))

## v0.2.4 (April 26th, 2024)

* Create a new version when model instance exist but no versions are ready yet ([#108](https://github.com/Kaggle/kagglehub/pull/108))
* Remove zipping in upload ([#105](https://github.com/Kaggle/kagglehub/pull/105))

## v0.2.3 (April 16th, 2024)

* Improve upload speed ([#100](https://github.com/Kaggle/kagglehub/pull/100))

## v0.2.2 (March 27th, 2024)

* Add support for single file upload ([#97](https://github.com/Kaggle/kagglehub/pull/97))

## v0.2.1 (March 21th, 2024)

* Add support for directory upload ([#82](https://github.com/Kaggle/kagglehub/pull/93))

## v0.2.0 (February 28th, 2024)

* Add raise_for_status() in post function ([#82](https://github.com/Kaggle/kagglehub/pull/89))
* Use Artifact Registry for helper images ([#83](https://github.com/Kaggle/kagglehub/pull/87))

## v0.1.9 (February 5th, 2023)

* Fix message when detecting newer version ([#82](https://github.com/Kaggle/kagglehub/pull/82))
* Link to model detail page in errors(Colab resolver) ([#83](https://github.com/Kaggle/kagglehub/pull/83))

## v0.1.8 (January 31st, 2024)

* Include URL to model detail page in error message ([#80](https://github.com/Kaggle/kagglehub/pull/80))
* Add Kaggle/Colab to user-agent if running on these environment ([#78](https://github.com/Kaggle/kagglehub/pull/78))
* Improve logging for Colab resolver ([#77](https://github.com/Kaggle/kagglehub/pull/77))

## v0.1.7 (January 29th, 2024)

* Fix `model_upload` with nested directory ([#75](https://github.com/Kaggle/kagglehub/pull/75))
* Detect if a newer version of `kagglehub` is available and suggest to upgrade ([#73](https://github.com/Kaggle/kagglehub/pull/73))

## v0.1.6 (January 22nd, 2024)

* Fix permission issue in `model_upload` and add integration tests ([#69](https://github.com/Kaggle/kagglehub/pull/69))
* Make specifying a license optional in `model_upload` ([#62](https://github.com/Kaggle/kagglehub/pull/62))
* Improve logging ([#68](https://github.com/Kaggle/kagglehub/pull/68), [#71](https://github.com/Kaggle/kagglehub/pull/71))
* Add resumable upload ([#55](https://github.com/Kaggle/kagglehub/pull/55))

## v0.1.5 (January 8th, 2024)

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
