"""
Tests for pyKES.pathways module.

This module tests the reaction pathway proportion calculation,
species propagation through reaction networks, and full network
propagation for photochemical systems.
"""

import pytest
from pyKES.reaction_ODE import parse_reactions
from pyKES.pathways.pathways import (
    calculate_reaction_pathway_proportions,
    propagate_species,
    merge_propagation_trees,
    calculate_reaction_network_propagation,
)


class TestCalculateReactionPathwayProportions:
    """Tests for calculate_reaction_pathway_proportions function."""

    def test_three_competing_pathways_equal_proportions(self):
        """
        Test pathway proportions for species [A] with three competing reactions.
        
        Reactions:
            [A] > [B], k1
            [A] + [C] > [D], k2
            [A] + [A] > [E], k3
        
        With [A]=0.1, [C]=0.5, k1=1.0, k2=2.0, k3=10.0:
            Rate1 = k1 * [A] = 1.0 * 0.1 = 0.1
            Rate2 = k2 * [A] * [C] = 2.0 * 0.1 * 0.5 = 0.1
            Rate3 = k3 * [A]^2 = 10.0 * 0.01 = 0.1
        All rates equal, so proportions are 1/3 each.
        """
        reactions = [
            '[A] > [B], k1',
            '[A] + [C] > [D], k2',
            '[A] + [A] > [E], k3'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 0.1, '[C]': 0.5}
        rate_constants = {'k1': 1.0, 'k2': 2.0, 'k3': 10.0}
        
        proportions = calculate_reaction_pathway_proportions(
            '[A]',
            concentrations,
            parsed_reactions,
            rate_constants
        )
        
        # All three pathways should have equal proportions (1/3)
        assert len(proportions) == 3
        assert proportions[0] == pytest.approx(1/3)
        assert proportions[1] == pytest.approx(1/3)
        assert proportions[2] == pytest.approx(1/3)
        
        # Proportions should sum to 1
        assert sum(proportions.values()) == pytest.approx(1.0)

    def test_species_with_no_reactions_returns_empty(self):
        """Test that a species with no reactions returns an empty dict."""
        reactions = ['[A] > [B], k1']
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 1.0}
        rate_constants = {'k1': 1.0}
        
        # [B] has no reactions where it's a reactant
        proportions = calculate_reaction_pathway_proportions(
            '[B]',
            concentrations,
            parsed_reactions,
            rate_constants
        )
        
        assert proportions == {}

    def test_zero_concentration_reactant_excluded(self):
        """Test that reactions with zero rate (due to zero co-reactant) are excluded."""
        reactions = [
            '[A] > [B], k1',
            '[A] + [C] > [D], k2'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[C]': 0.0}  # [C] is zero
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        proportions = calculate_reaction_pathway_proportions(
            '[A]',
            concentrations,
            parsed_reactions,
            rate_constants
        )
        
        # Only reaction 0 should have non-zero rate
        assert len(proportions) == 1
        assert 0 in proportions
        assert proportions[0] == pytest.approx(1.0)

    def test_dominant_pathway(self):
        """Test proportions when one pathway dominates."""
        reactions = [
            '[A] > [B], k1',
            '[A] + [C] > [D], k2'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[C]': 100.0}  # High [C]
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        proportions = calculate_reaction_pathway_proportions(
            '[A]',
            concentrations,
            parsed_reactions,
            rate_constants
        )
        
        # Reaction 1 should dominate (rate = 100 vs rate = 1)
        assert proportions[0] == pytest.approx(1/101)
        assert proportions[1] == pytest.approx(100/101)


class TestMergePropagationTrees:
    """Tests for merge_propagation_trees function."""

    def test_simple_merge_amounts(self):
        """Test that amounts are correctly summed during merge."""
        tree1 = {'amount_formed': 0.5}
        tree2 = {'amount_formed': 0.3}
        
        merged = merge_propagation_trees(tree1, tree2)
        
        assert merged['amount_formed'] == pytest.approx(0.8)

    def test_merge_with_nested_species(self):
        """Test merging trees with nested species."""
        tree1 = {
            'amount_formed': 0.5,
            '[B]': {'amount_formed': 0.25}
        }
        tree2 = {
            'amount_formed': 0.3,
            '[B]': {'amount_formed': 0.15}
        }
        
        merged = merge_propagation_trees(tree1, tree2)
        
        assert merged['amount_formed'] == pytest.approx(0.8)
        assert merged['[B]']['amount_formed'] == pytest.approx(0.4)

    def test_merge_with_different_species(self):
        """Test merging trees with different species keys."""
        tree1 = {
            'amount_formed': 0.5,
            '[B]': {'amount_formed': 0.25}
        }
        tree2 = {
            'amount_formed': 0.3,
            '[C]': {'amount_formed': 0.15}
        }
        
        merged = merge_propagation_trees(tree1, tree2)
        
        assert merged['amount_formed'] == pytest.approx(0.8)
        assert merged['[B]']['amount_formed'] == pytest.approx(0.25)
        assert merged['[C]']['amount_formed'] == pytest.approx(0.15)


class TestPropagateSpecies:
    """Tests for propagate_species function."""

    def test_two_pathways_separate_products(self):
        """
        Test propagation where two pathways lead to different products.
        
        Reactions:
            [A] > [B] + 2 [D], k1
            [A] + [A] > [C], k2
        
        With equal rate constants and [A]=1.0:
            Rate1 = k1 * [A] = 1.0
            Rate2 = k2 * [A]^2 = 1.0
        Proportions: 50% each pathway.
        """
        reactions = [
            '[A] > [B] + 2 [D], k1',
            '[A] + [A] > [C], k2'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 0.0, '[C]': 0.0, '[D]': 0.0}
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        assert tree['[B]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[D]']['amount_formed'] == pytest.approx(1.0)  # 2 * 0.5
        assert tree['[C]']['amount_formed'] == pytest.approx(0.5)

    def test_two_pathways_same_product_merging(self):
        """
        Test propagation where two pathways lead to the same product [B].
        
        Reactions:
            [A] > [B] + 2 [D], k1
            [A] + [A] > [B], k2
        
        [B] is formed from both pathways and should be merged.
        """
        reactions = [
            '[A] > [B] + 2 [D], k1',
            '[A] + [A] > [B], k2'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 0.0, '[D]': 0.0}
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        # [B] formed from both pathways: 0.5 + 0.5 = 1.0
        assert tree['[B]']['amount_formed'] == pytest.approx(1.0)
        assert tree['[D]']['amount_formed'] == pytest.approx(1.0)

    def test_species_formed_at_different_levels_not_merged(self):
        """
        Test that [D] formed at different levels in the tree is not incorrectly merged.
        
        Reactions:
            [A] > [B] + 2 [D], k1
            [B] > [D], k2
        
        [D] is formed directly from [A] and also as a downstream product from [B].
        These should appear at different levels in the tree.
        """
        reactions = [
            '[A] > [B] + 2 [D], k1',
            '[B] > [D], k2'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 1.0, '[D]': 0.0}
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        # [B] is formed with amount 1.0 (stoichiometry 1)
        assert tree['[B]']['amount_formed'] == pytest.approx(1.0)
        # [D] at top level: 2.0 (stoichiometry 2)
        assert tree['[D]']['amount_formed'] == pytest.approx(2.0)
        # [D] nested under [B]: 1.0
        assert tree['[B]']['[D]']['amount_formed'] == pytest.approx(1.0)

    def test_diamond_pattern_separate_branches(self):
        """
        Test diamond pattern where [D] is reached via [B] and [C] separately.
        
        Reactions:
            [A] > [B], k1
            [A] > [C], k2
            [B] > [D], k3
            [C] > [D], k4
            [D] > [F], k5
        
        [D] is formed via two independent branches, each with its own subtree.
        """
        reactions = [
            '[A] > [B], k1',
            '[A] > [C], k2',
            '[B] > [D], k3',
            '[C] > [D], k4',
            '[D] > [F], k5'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {
            '[A]': 1.0, '[B]': 1.0, '[C]': 1.0, '[D]': 1.0, '[F]': 0.0
        }
        rate_constants = {
            'k1': 1.0, 'k2': 1.0, 'k3': 1.0, 'k4': 1.0, 'k5': 1.0
        }
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        # 50% goes to [B], 50% to [C]
        assert tree['[B]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[C]']['amount_formed'] == pytest.approx(0.5)
        # Each branch has [D] -> [F]
        assert tree['[B]']['[D]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[B]']['[D]']['[F]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[C]']['[D]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[C]']['[D]']['[F]']['amount_formed'] == pytest.approx(0.5)

    def test_complex_with_markers_recursive_merging(self):
        """
        Test complex reaction network with markers requiring recursive merging.
        
        Reactions:
            [A] > [B] + [Marker-0], k1
            [A] + [B] > 2 [B] + [Marker-1], k2
            [B] > [C] + [Marker-2], k3
            [B] > [D] + [Marker-3], k4
            [C] > [E] + [Marker-4], k5
        
        This tests recursive merging when [B] is formed from multiple pathways
        and propagates downstream.
        """
        reactions = [
            '[A] > [B] + [Marker-0], k1',
            '[A] + [B] > 2 [B] + [Marker-1], k2',
            '[B] > [C] + [Marker-2], k3',
            '[B] > [D] + [Marker-3], k4',
            '[C] > [E] + [Marker-4], k5'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 1.0, '[C]': 1e-9, '[D]': 1e-9}
        rate_constants = {
            'k1': 1.0, 'k2': 1.0, 'k3': 1.0, 'k4': 1.0, 'k5': 1.0
        }
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        # [B] formed with merged amounts: 1.5 (0.5 from k1 + 1.0 from k2)
        assert tree['[B]']['amount_formed'] == pytest.approx(1.5)
        # [Marker-0] from pathway 1
        assert tree['[Marker-0]']['amount_formed'] == pytest.approx(0.5)
        # [Marker-1] appears both at top level and nested
        assert tree['[Marker-1]']['amount_formed'] == pytest.approx(0.5)
        # Downstream from [B]
        assert tree['[B]']['[C]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[B]']['[D]']['amount_formed'] == pytest.approx(0.5)
        assert tree['[B]']['[C]']['[E]']['amount_formed'] == pytest.approx(0.5)

    def test_cycle_prevention(self):
        """Test that cycles in reaction networks are properly handled."""
        reactions = [
            '[A] > [B], k1',
            '[B] > [A], k2'  # Cycle back to [A]
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 1.0}
        rate_constants = {'k1': 1.0, 'k2': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        # Should complete without infinite recursion
        assert tree['amount_formed'] == pytest.approx(1.0)
        assert tree['[B]']['amount_formed'] == pytest.approx(1.0)
        # [A] from [B] should stop propagation (cycle)
        assert tree['[B]']['[A]']['amount_formed'] == pytest.approx(1.0)
        # No further propagation from the cycled [A]
        assert '[B]' not in tree['[B]']['[A]']

    def test_terminal_species_no_further_reactions(self):
        """Test that terminal species (no further reactions) are handled correctly."""
        reactions = ['[A] > [B], k1']
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 0.0}
        rate_constants = {'k1': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        assert tree['[B]']['amount_formed'] == pytest.approx(1.0)
        # [B] should only have amount_formed, no further reactions
        assert len(tree['[B]']) == 1


class TestCalculateReactionNetworkPropagation:
    """Integration tests for calculate_reaction_network_propagation function."""

    def test_photochemical_network_propagation(self):
        """
        Test full photochemical reaction network propagation.
        
        This tests the complete workflow from light absorption through
        excited state formation and downstream reactions.
        """
        photon_flux = 1e17
        pathlength = 2.25
        
        extinction_coefficients = {
            '[A]': 8500,
            '[B]': 5400,
            '[C]': 1000
        }
        
        reactions = [
            '[A] > [A-excited], k1',
            '[A-excited] > [A], k2',
            '[A-excited] > [B], k3',
            '[B] > [A], k4',
            '[B] > [B-excited], k5',
            '[B-excited] > [B], k6',
            '[B-excited] > [C], k7'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        rate_constants = {
            'k1': 1,
            'k2': 3e8,
            'k3': 1e8,
            'k4': 1e2,
            'k5': 1,
            'k6': 2e8,
            'k7': 3.3e8
        }
        
        concentrations = {
            '[A]': 5.0,
            '[A-excited]': 0.001,
            '[B]': 5.0,
            '[B-excited]': 0.001,
            '[C]': 1.0
        }
        
        absorbing_species = {
            '[A]': '[A-excited]',
            '[B]': '[B-excited]',
            '[C]': '[C-excited]'
        }
        
        result = calculate_reaction_network_propagation(
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            absorbing_species=absorbing_species,
            extinction_coefficients=extinction_coefficients,
            photon_flux=photon_flux,
            pathlength=pathlength
        )
        
        # Check structure
        assert 'Light absorption' in result
        light_abs = result['Light absorption']
        
        # Check transmitted fraction
        assert light_abs['transmitted'] == pytest.approx(0.6940248165645054, rel=1e-6)
        
        # Check [A] absorption and propagation
        assert '[A]' in light_abs
        assert light_abs['[A]']['absorbed'] == pytest.approx(0.18445312476607828, rel=1e-6)
        
        a_excited = light_abs['[A]']['[A-excited]']
        assert a_excited['amount_formed'] == pytest.approx(0.18445312476607828, rel=1e-6)
        
        # Check [A-excited] deactivation back to [A] (75% based on k2/(k2+k3))
        assert a_excited['[A]']['amount_formed'] == pytest.approx(0.1383398435745587, rel=1e-6)
        
        # Check [A-excited] -> [B] (25%)
        assert a_excited['[B]']['amount_formed'] == pytest.approx(0.04611328119151957, rel=1e-6)
        
        # Check [B] absorption and propagation
        assert '[B]' in light_abs
        assert light_abs['[B]']['absorbed'] == pytest.approx(0.11718198514550855, rel=1e-6)
        
        b_excited = light_abs['[B]']['[B-excited]']
        assert b_excited['amount_formed'] == pytest.approx(0.11718198514550855, rel=1e-6)
        
        # [B-excited] splits to [B] and [C] based on k6 and k7
        assert b_excited['[B]']['amount_formed'] == pytest.approx(0.044219617036040965, rel=1e-6)
        assert b_excited['[C]']['amount_formed'] == pytest.approx(0.07296236810946759, rel=1e-6)
        
        # Check [C] absorption
        assert '[C]' in light_abs
        assert light_abs['[C]']['absorbed'] == pytest.approx(0.004340073523907725, rel=1e-6)

    def test_pathway_proportions_in_photochemical_network(self):
        """Test pathway proportions for [B] in the photochemical network."""
        reactions = [
            '[A] > [A-excited], k1',
            '[A-excited] > [A], k2',
            '[A-excited] > [B], k3',
            '[B] > [A], k4',
            '[B] > [B-excited], k5',
            '[B-excited] > [B], k6',
            '[B-excited] > [C], k7'
        ]
        parsed_reactions, species = parse_reactions(reactions)
        
        rate_constants = {
            'k1': 1,
            'k2': 3e8,
            'k3': 1e8,
            'k4': 1e2,
            'k5': 1,
            'k6': 2e8,
            'k7': 3.3e8
        }
        
        concentrations = {
            '[A]': 5.0,
            '[A-excited]': 0.001,
            '[B]': 5.0,
            '[B-excited]': 0.001,
            '[C]': 1.0
        }
        
        proportions = calculate_reaction_pathway_proportions(
            '[B]',
            concentrations,
            parsed_reactions,
            rate_constants
        )
        
        # [B] reacts via k4 ([B] > [A]) and k5 ([B] > [B-excited])
        # Rate4 = k4 * [B] = 100 * 5 = 500
        # Rate5 = k5 * [B] = 1 * 5 = 5
        # Total = 505
        assert proportions[3] == pytest.approx(0.9900990099009901, rel=1e-6)  # k4 pathway
        assert proportions[4] == pytest.approx(0.009900990099009901, rel=1e-6)  # k5 pathway


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_reactions(self):
        """Test with a species that has no reactions."""
        reactions = ['[A] > [B], k1']
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 1.0, '[C]': 1.0}
        rate_constants = {'k1': 1.0}
        
        # Start from a species not in any reaction
        tree = propagate_species(
            species='[C]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        # Should return just the amount_formed
        assert tree == {'amount_formed': 1.0}

    def test_zero_amount_propagation(self):
        """Test propagation with zero amount."""
        reactions = ['[A] > [B], k1']
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 0.0}
        rate_constants = {'k1': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=0.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(0.0)
        assert tree['[B]']['amount_formed'] == pytest.approx(0.0)

    def test_stoichiometry_greater_than_one(self):
        """Test that stoichiometry coefficients are properly applied."""
        reactions = ['[A] > 3 [B], k1']
        parsed_reactions, species = parse_reactions(reactions)
        
        concentrations = {'[A]': 1.0, '[B]': 0.0}
        rate_constants = {'k1': 1.0}
        
        tree = propagate_species(
            species='[A]',
            amount=1.0,
            concentrations=concentrations,
            parsed_reactions=parsed_reactions,
            rate_constants=rate_constants,
            other_multipliers={},
            ancestor_chain=set()
        )
        
        assert tree['amount_formed'] == pytest.approx(1.0)
        # 3 [B] formed per [A] consumed
        assert tree['[B]']['amount_formed'] == pytest.approx(3.0)
