# FAQ and Known Issues

## FAQ

### What should I do if I see unexpected behaviors of the uploader?

Please reload the web app first. If the issue persists, please let us know.

### What is the definition of a "night"?

A night used for the visibility check starts at 18:30 and ends at 5:30 on next day.
This is subject to change.

### What is the definition of "fiberhour"?

For example, one fiberhour corresponds to observing 1 objects for 1 hour.
It is actually equivalent to 4 objects for 15 minutes.

### I forgot to record my `Upload ID`. What can I do?

See the note in [the `Submission` page](submission.md#upload-id).

## Known Issues

### Result of pointing simulation varies with the identical input targets

Because the fiber assignment is a non-linear problem, there is some randomness involved in the solution.

### Pointing simulation does not seem to finish

It is a known issue that the plotting library (`hvplot`) cannot render more than 4000 polygons.
If your target list is large and/or exposure time is long, please consider to reduce them.

In general, computational time is long, but sometimes it freezes due to the excessive use of memory, web server's timeout, etc.
You can try to reload and start the simulation again.  If you are not sure what's going on, please contact us.

Note that the computation will be terminated once the running time reaches 15 minutes.

### Large `obj_id` in the tables is not displayed correctly in tables

Javascript is used to display the table and the limitation of integer is $2^{53}-1$ ([ref](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Number/MAX_SAFE_INTEGER)). Numbers larger than the limit are incorrectly displayed.

### Tables are not displayed correctly

Sometimes tables are not rendered correctly. If your table has more than one page, moving to another page usually bring the content back for the first page.
In the case of a single page table, we are still not clear how to recover the content. You can run the validation and simulation multiple times, which usually solve the issue for some reason.
