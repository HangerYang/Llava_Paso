## Multimodal Project Github

Please be aware that this repo is somewhat messy at the moment. I have been playing around with code from various online repos and combining
them in (hopefully) useful ways, not worrying too much about good practices.

## Key Parts

## 1: MIR-LLaVA

This is a copy of the LLaVA repo with my additions from the MIR repo to check MIR. Originally I tried to work within the MIR repo but I had issues with their code
so I combined it with my already-working LLaVA repo.

### Key file: MIRmeasurement.ipynb

Run this file to measure the MIR of any llava models. Note that you will need to download the data from the MIR repo for this to work.

This file loads in a model then called the eval_model2 function, which measures the MIR of the llava-based model. Again, sorry for the poor naming, I've been
playing around to just find parts that work.

### Update:

In MIRmeasurement.ipynb, I download the pretrained projector weights and used the base vicuna LLM as input
to the function that loads the weights. There is probably a better way to do this but I think this works for now.

## 2: beavertails

This is a copy of the beavertails repo with some modifications I was using to evaluate model outputs.

### Key file: examples/moderation/batch_evaluate_booster.sh

This file runs batch_evaluate_booster.py, which uses the moderation model from beavertails to evaluate the target model outputs. Notice that the prompt
and response pairs from the target model must be stored in a json file similar to the one in the same folder as the script titled
llava15_mocha__outputs_grenadebomb.json

## Final thoughts

Again, this repo doesn't have good coding practices, it just has what I've been playing around with recently. As for the virtual environments, I have been using
the ones I created from the LLaVA and beavertails repos, respectively.
