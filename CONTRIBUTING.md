# How to Contribute

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code Reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

### Plugin and Skill Guidelines

-   Place general, robot-agnostic capabilities under `plugins/robotics/skills/`
    and platform-specific bring-up, kinematics, or hardware parameters under
    `plugins/<platform>/skills/`.
-   Ensure every skill folder name matches the `name:` field in its `SKILL.md`
    YAML frontmatter.
-   Include the standard Apache 2.0 `Copyright 2026 Google LLC` header at the
    top of any new `.py` or `.sh` script.

## Community Guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).
