/**
 * Unit tests for summing matrix construction.
 *
 * Pure logic — no model required. Uses structural hierarchy definitions
 * (the topology itself is the test fixture; arithmetic assertions use
 * deterministic synthetic values within test scope).
 */

import { describe, it, expect } from 'vitest';
import { buildSummingMatrix } from '../src/summing-matrix';
import type { HierarchyDefinition } from '../src/types';

// ---------------------------------------------------------------------------
// Helper: standard 3-level hierarchy (Total → {West, East} → 4 stores)
// ---------------------------------------------------------------------------

const THREE_LEVEL_HIERARCHY: HierarchyDefinition = {
  nodes: [
    { id: 'total', parentId: null },
    { id: 'west', parentId: 'total' },
    { id: 'east', parentId: 'total' },
    { id: 's1', parentId: 'west' },
    { id: 's2', parentId: 'west' },
    { id: 's3', parentId: 'east' },
    { id: 's4', parentId: 'east' },
  ],
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('buildSummingMatrix', () => {
  // ── Structure ──────────────────────────────────────────────────────────

  it('produces a 7×4 S matrix for a 7-node / 4-leaf hierarchy', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    expect(result.S.length).toBe(7); // m = total nodes
    expect(result.S[0].length).toBe(4); // n = bottom nodes
  });

  it('orders allNodeIds topologically (parents before children)', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    const totalIdx = result.allNodeIds.indexOf('total');
    const westIdx = result.allNodeIds.indexOf('west');
    const eastIdx = result.allNodeIds.indexOf('east');
    const s1Idx = result.allNodeIds.indexOf('s1');

    expect(totalIdx).toBeLessThan(westIdx);
    expect(totalIdx).toBeLessThan(eastIdx);
    expect(westIdx).toBeLessThan(s1Idx);
  });

  it('returns bottomNodeIds sorted alphabetically', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    expect(result.bottomNodeIds).toEqual(['s1', 's2', 's3', 's4']);
  });

  // ── S-matrix correctness ───────────────────────────────────────────────

  it('fills S[row=total] with all ones (aggregates all bottom nodes)', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    const totalIdx = result.allNodeIds.indexOf('total');
    expect(result.S[totalIdx]).toEqual([1, 1, 1, 1]);
  });

  it('fills S[row=west] with [1,1,0,0] (aggregates west bottom nodes only)', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    const westIdx = result.allNodeIds.indexOf('west');
    expect(result.S[westIdx]).toEqual([1, 1, 0, 0]);
  });

  it('fills S[row=east] with [0,0,1,1]', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    const eastIdx = result.allNodeIds.indexOf('east');
    expect(result.S[eastIdx]).toEqual([0, 0, 1, 1]);
  });

  it('fills S[row=leaf] with a one-hot row (only itself)', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    const s1Idx = result.allNodeIds.indexOf('s1');
    expect(result.S[s1Idx]).toEqual([1, 0, 0, 0]);
  });

  // ── Coherence round-trip ───────────────────────────────────────────────

  it('reproduces ancestor values via summing matrix (coherence property)', () => {
    const result = buildSummingMatrix(THREE_LEVEL_HIERARCHY);
    // Given bottom-level values [10, 20, 30, 40] at [s1, s2, s3, s4]
    const bottomVals = [10, 20, 30, 40];

    // Each node's aggregate = Σ S[row][col] * bottomVals[col]
    for (let i = 0; i < result.allNodeIds.length; i++) {
      let sum = 0;
      for (let j = 0; j < result.bottomNodeIds.length; j++) {
        sum += result.S[i][j] * bottomVals[j];
      }

      const id = result.allNodeIds[i];
      switch (id) {
        case 'total':
          expect(sum).toBe(100);
          break; // 10+20+30+40
        case 'west':
          expect(sum).toBe(30);
          break; // 10+20
        case 'east':
          expect(sum).toBe(70);
          break; // 30+40
        case 's1':
          expect(sum).toBe(10);
          break;
        case 's2':
          expect(sum).toBe(20);
          break;
        case 's3':
          expect(sum).toBe(30);
          break;
        case 's4':
          expect(sum).toBe(40);
          break;
      }
    }
  });

  // ── Single-node hierarchy ──────────────────────────────────────────────

  it('handles a single-node hierarchy (root is also leaf)', () => {
    const single: HierarchyDefinition = {
      nodes: [{ id: 'only', parentId: null }],
    };
    const result = buildSummingMatrix(single);
    expect(result.S.length).toBe(1);
    expect(result.S[0].length).toBe(1);
    expect(result.S[0][0]).toBe(1);
    expect(result.allNodeIds).toEqual(['only']);
    expect(result.bottomNodeIds).toEqual(['only']);
  });

  // ── Two-level hierarchy ────────────────────────────────────────────────

  it('handles a 2-level hierarchy (Total → 3 children)', () => {
    const twoLevel: HierarchyDefinition = {
      nodes: [
        { id: 'total', parentId: null },
        { id: 'a', parentId: 'total' },
        { id: 'b', parentId: 'total' },
        { id: 'c', parentId: 'total' },
      ],
    };
    const result = buildSummingMatrix(twoLevel);
    expect(result.S.length).toBe(4);
    expect(result.S[0].length).toBe(3);
    expect(result.S[0]).toEqual([1, 1, 1]); // total
  });

  // ── Validation errors ──────────────────────────────────────────────────

  it('throws on empty hierarchy', () => {
    expect(() => buildSummingMatrix({ nodes: [] })).toThrow('at least one node');
  });

  it('throws on duplicate node ids', () => {
    const dup: HierarchyDefinition = {
      nodes: [
        { id: 'root', parentId: null },
        { id: 'root', parentId: null },
      ],
    };
    expect(() => buildSummingMatrix(dup)).toThrow('Duplicate node id');
  });

  it('throws on multiple roots', () => {
    const multiRoot: HierarchyDefinition = {
      nodes: [
        { id: 'r1', parentId: null },
        { id: 'r2', parentId: null },
      ],
    };
    expect(() => buildSummingMatrix(multiRoot)).toThrow('Multiple root');
  });

  it('throws on no root', () => {
    const noRoot: HierarchyDefinition = {
      nodes: [
        { id: 'a', parentId: 'b' },
        { id: 'b', parentId: 'a' },
      ],
    };
    expect(() => buildSummingMatrix(noRoot)).toThrow('No root');
  });

  it('throws on orphan (parentId not in hierarchy)', () => {
    const orphan: HierarchyDefinition = {
      nodes: [
        { id: 'root', parentId: null },
        { id: 'child', parentId: 'missing' },
      ],
    };
    expect(() => buildSummingMatrix(orphan)).toThrow('unknown parent');
  });

  it('throws when a cycle creates unreachable nodes', () => {
    const withCycle: HierarchyDefinition = {
      nodes: [
        { id: 'a', parentId: null },
        { id: 'b', parentId: 'c' },
        { id: 'c', parentId: 'b' },
        { id: 'd', parentId: 'a' },
      ],
    };
    expect(() => buildSummingMatrix(withCycle)).toThrow('Unreachable');
  });
});
