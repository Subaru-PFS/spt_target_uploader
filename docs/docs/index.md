# Welcome

The PFS Target Uploader is a web app to and submit the target list supplied by users with time estimate by a pointing simulation.

## Workflow

```mermaid
graph TD
  select_file[Select an input target list] --> validate[Validation of the target list];
  validate -->|Success| run_ppp[Total Exposure Time Estimate];
  validate -->|Fail| fix_errors[Fix the target list];
  fix_errors --> select_file;
  run_ppp -->|Happy| submit_results[Submission of the target and pointing lists];
  run_ppp -->|Unhappy| fix_errors;
  submit_results --> done[Done];
```

## [Input Target List](inputs.md)

- [File format](inputs.md#file-format)
- [Content of the list](inputs.md#content)

## [Validation](validation.md)

- [Validation stages](validation.md#stages)
- [Understand the results](validation.md#results)

## [Total Exposure Time Estimate](PPP.md)

- [Run](PPP.md#run)
- [Understand the results](PPP.md#results)

## [Submission](submission.md)

- [Submit target list](submission.md#submit-the-target-list)
- [Submit PPP outputs](submission.md#submit-the-ppp-outputs)

## [FAQ & Known Issues](issues.md)

- [FAQ](issues.md#faq)
- [Known Issues](issues.md#known-issues)


## [Contact](contact.md)

Any feedback is welcome. Please contact Masato Onodera and Wanqiu He (Subaru Telescope, NAOJ) via PFS Slack and/or email (<monodera@naoj.org>, <wanqiu.he@nao.ac.jp>).
