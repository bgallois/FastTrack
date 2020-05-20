# Contributing

Please first discuss the change you wish to make via email (benjamin.gallois@fasttrack.sh) before contributing to the Fast Track project. Please note we have a code of conduct, please follow it.

## Installation

Before contributing to Fast Track, make sure to have a working environment of development, [see the developer guide](http://http://www.fasttrack.sh/API/index.html).

## Contributing

### Create a branch

Before any work can be done, you have to create a new branch. If you are working on a new feature, branch the dev branch. For a bug fix, branch the master branch.

#### Branch naming convention

The branch must always follow this naming convention:
```name/type/description```
With:
* name: gitlab name
* type: keywords among the commit type list (see below)
* description: short description in lower case, maximum 4 words separated by a hyphen

```
$ git checkout master
$ git pull master
$ git push origin master
$ git checkout -b bgallois/docs/update-contributing
```

```
$ git checkout dev
$ git pull dev
$ git push origin dev
$ git checkout -b bgallois/fix/fix-crash-crop
```


### Create a commit

#### Commit naming convention

Each commit must follow this naming convention to ensure the readability of the commit history:
```
[type] scope: description
```

##### Commit type

* `build`: Changes in build and deployment scripts.
* `docs`: Changes in the documentation.
* `feat`: Implementation of new functionality.
* `fix`: Fix a bug.
* `opti`: Performance optimization.
* `ref`: Code refactoring.
* `style`: User interface refactoring.
* `rel`: Release, note that they are no scope in these commits.
* `test`: Changes in the tests.
* `wip`: Saved work in progress, temporary commit not authorized in the final merge.

##### Commit scope

* `global`: All the project.
* `interactive`: Changes in the interactive tracking class.
* `replay`: Changes in the replay class.
* `batch`: Changes in the batch class.
* `main`: Changes in the mainwindow class.
* `tracking`: Changes in the tracking class.

The scope is determined by the modification made in the commit, the scope is always the part of the code that has required to create a new commit. For example, a new feature in the interactive class that requires a modification in the tracking class has an interactive scope. A bug that appears in the interactive class but that is caused by an error in the tracking class has a tracking scope.

### Formatting

Fast Track is using [Clang format](https://clang.llvm.org/docs/ClangFormat.html) to ensure the homogeneity of code formatting inside the project.
* Format your code according to the .clang-format file.
* Format your documentation following the [Doxygen](http://doxygen.nl/) formatting.
* Follow a standard C++ [naming convention](https://google.github.io/styleguide/cppguide.html#Naming).  
### Test

Fast Track is using [Google Test framework](https://github.com/google/googletest) to perform unitary tests.
* Each commit has to pass all the tests, to run the test:
```
$ ./test.sh
```
* If you add some features to the tracking algorithm, you have to implement tests. New features to the tracking algorithm without tests will be refused.
* New features that are purely on the graphical interface can omit tests.
* 
### Create a merge request

Before creating the merge request, make sure that you follow the project guidelines. Pull request that will not follow all the guidelines will be automatically refused.
```
git push -u origin bgallois/docs/update-contributing
```

```
$ git push -u origin -b bgallois/fix/fix-crash-crop
```

Go to the [project page](https://git.benjamin-gallois.fr/bgallois/FastTrack) and click on the **Merge Request** tab. Click on **New merge request**. Select the source branch and the target branch and click on **Compare branches and continue** button. Finally, click on the **Submit merge request**.

[More information](https://docs.gitlab.com/ee/gitlab-basics/add-merge-request.html).

## Our Responsibilities

Project maintainers are responsible for clarifying the standards of acceptable behavior and are expected to take appropriate and fair corrective action in response to any instances of unacceptable behavior.

Project maintainers have the right and responsibility to remove, edit, or reject comments, commits, code, wiki edits, issues, and other contributions that are not aligned to this Code of Conduct, or to ban temporarily or permanently any contributor for other behaviors that they deem inappropriate, threatening, offensive, or harmful.
