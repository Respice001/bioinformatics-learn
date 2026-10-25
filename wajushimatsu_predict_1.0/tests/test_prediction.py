import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from predictor import (load_public_model, trait, predict, founder_posterior,
                       parent_posterior, make_synthetic_model)
from mendel import sex_transition


class PublicInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=load_public_model()

    def test_public_no_individual_or_private_records(self):
        public=self.model
        self.assertNotIn('birds',public)
        self.assertNotIn('posterior',public)
        self.assertEqual(public['counts']['records'],785)
        self.assertEqual(set(public['loci']),{'rpd','ch','fwn','pl'})
        for v in public['loci'].values():
            self.assertNotIn('posterior',v)
            self.assertNotIn('birds',v)

    def test_all_locus_priors_normalize_and_z_female_no_het(self):
        for loc,p in self.model['loci'].items():
            for sex in ('M','F'):
                g=founder_posterior(trait(),sex,loc,p)
                self.assertAlmostEqual(sum(g),1.0,places=12)
                if loc=='pl' and sex=='F':
                    self.assertAlmostEqual(g[1],0,places=12)

    def test_unknown_grandparents_match_population_prior(self):
        for loc,p in self.model['loci'].items():
            for sex in ('M','F'):
                base=parent_posterior(trait(),sex,loc,p)
                with_grandparents=parent_posterior(trait(),sex,loc,p,(trait(),trait()))
                self.assertTrue(np.allclose(base,with_grandparents,rtol=1e-10,atol=1e-10))

    def test_grandparent_phenotype_informs_hidden_carrier(self):
        p=self.model['loci']['ch']
        parent=trait('和黒','通常',0)
        baseline=parent_posterior(parent,'M','ch',p)
        augmented=parent_posterior(parent,'M','ch',p,(trait('ハバナ'),trait('和黒')))
        self.assertGreater(augmented[1],baseline[1])

    def test_autosomal_transmission(self):
        a=sex_transition([0,1,0],[0,1,0],'autosomal','?')
        self.assertTrue(np.allclose(a,[.25,.5,.25]))

    def test_z_linked_mother_daughter(self):
        # father mutant Z/Z and normal mother Z+/W -> all daughters mutant Z/W;
        # all sons heterozygous carriers.
        father=[0,0,1];mother=[1,0,0]
        d=sex_transition(father,mother,'z_linked','F')
        s=sex_transition(father,mother,'z_linked','M')
        self.assertTrue(np.allclose(d,[0,0,1]))
        self.assertTrue(np.allclose(s,[0,1,0]))

    def test_pred_joint_probability_and_bands(self):
        r=predict(self.model,trait('和黒','通常',0),trait('ハバナ','通常',1),draws=150)
        values=r['all_combinations']
        self.assertGreater(len(values),15)
        self.assertAlmostEqual(sum(v['probability'] for v in values),1,places=3)
        self.assertTrue(all(0<=v['ci95_low']<=v['probability']<=v['ci95_high']<=1 for v in values))
        self.assertFalse(any(v['base'] in ('アルビノ','白') and v['pied'] for v in values))
        self.assertTrue(any(v['ci95_high']-v['ci95_low']>.001 for v in values))

    def test_predictions_are_deterministic(self):
        a=predict(self.model,trait('和黒'),trait('ハバナ'),draws=100,seed=777)
        b=predict(self.model,trait('和黒'),trait('ハバナ'),draws=100,seed=777)
        self.assertEqual(a['all_combinations'],b['all_combinations'])

    def test_ancestors_change_prediction(self):
        f,m=trait('和黒','通常',0),trait('ハバナ','通常',0)
        a=predict(self.model,f,m,draws=150)
        b=predict(self.model,f,m,grandparents={'father':(trait('ハバナ'),trait('和黒'))},draws=150)
        self.assertGreater(b['genotype_posteriors']['ch']['father'][1],a['genotype_posteriors']['ch']['father'][1])
        self.assertNotEqual(a['top_combinations'][0]['probability'],b['top_combinations'][0]['probability'])

    def test_white_is_not_interpreted_as_no_pied(self):
        self.assertIsNone(trait('白','通常',0)['pied'])
        self.assertIsNone(trait('アルビノ','薄',1)['shade'])

if __name__=='__main__':
    unittest.main()
