# Upstream

Installed from [nateherkai/herk-edit-kit](https://github.com/nateherkai/herk-edit-kit)
at commit `10511e5` ("Playbook: cut between phrases, never inside one").

Left out to keep this repository small (about 400 MB of media):

- `video-projects/`: the 12 original teaching projects (the upstream `.gitignore` already ignores this folder)
- `examples/showcase/*.mp4`: the four showcase videos linked from the README

Local change: `scripts/check-kit.mjs` skips `.remember/`.

To update, clone upstream again and copy it over this repository, leaving out the same paths.
