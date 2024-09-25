# FAQ and Known Issues

## FAQ

### What should I do if I see unexpected behaviors of the uploader?

Please reload the web app first. If the issue persists, please let us know.

### What is the definition of a "night"?

A night used for the visibility check starts at 18:30 and ends at 5:30 on the next day.
This is subject to change.

### What is the definition of "fiberhour"?

For example, one fiberhour corresponds to observing 1 object for 1 hour.
It is actually equivalent to 4 objects for 15 minutes.

### I forgot to record my `Upload ID`. What can I do?

See the note on [the `Submission` page](submission.md#upload-id).

## Known Issues

### The result of the pointing simulation varies with the identical input targets

Because the fiber assignment is a non-linear problem, there is some randomness involved in the solution.

### Pointing simulation does not seem to finish

There are some cases for which the pointing simulation seems to stuck.
This includes the cases such as plotting more than 4000 polygons (apparently `hvplot` issue),
using Firefox with many pointings, using the excessive use of memory, web server's timeout, etc.

For troubleshoot, consider the following:

- Please reload and re-start the simulation.
- If your target list is large and/or exposure time is long, consider reducing them.
- If you are using Firefox, try with different browsers using other than the Gecko engine.

Note that the computation may be terminated once the running time reaches 15 minutes or the number of pointings exceeds 200 in a given resolution mode.

If you are not sure what's going on, please contact us.

### Tables are not displayed correctly

Sometimes tables are not rendered correctly. If your table has more than one page, moving to another page usually brings the content back to the first page.
In the case of a single-page table, we are still not clear on how to recover the content. You can run the validation and simulation multiple times, which usually solves the issue for some reason.

### The input file is not read properly after modification

Workarounds can be changing the filename, switching to a non-WebKit browser such as Firefox, or reloading the app to proceed with your modified file.
If you use Google Chrome or WebKit-based browsers, you may encounter an issue that the content of the input file is not properly loaded and displayed after some modification of the content following validation errors and/or simulation results. It seems a WebKit-related issue and we have not found a solid solution to this.
