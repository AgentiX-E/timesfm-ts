/**
 * Summing matrix construction for hierarchical time series.
 *
 * The summing matrix S (m × n) maps bottom-level forecasts to all nodes:
 *   S[i][j] = 1 iff bottom node j contributes to aggregate node i.
 *
 * For any hierarchy with m total nodes and n bottom-level (leaf) nodes,
 * the reconciling equation is: ŷ_h = S · P · ŷ_b
 * where ŷ_b is the length-m vector of base forecasts and P is the
 * n × m projection matrix (computed by reconciliation.ts).
 *
 * Reference: Hyndman et al. (2011), §2.1.
 */

import type { HierarchyDefinition, HierarchyNode } from './types';

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

export interface SummingMatrixResult {
  /** S (m × n): S[i][j] = 1 iff bottom node j contributes to node i. */
  readonly S: number[][];
  /** Node ids in row order (length m) — topologically sorted, parents before children. */
  readonly allNodeIds: readonly string[];
  /** Bottom-level node ids in column order (length n). */
  readonly bottomNodeIds: readonly string[];
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/**
 * Look up a value in a Map, throwing a descriptive error when the key
 * is missing instead of using a non-null assertion.  All call-sites in
 * this module are guaranteed to succeed because keys are validated
 * beforehand, but this keeps the type-checker and linter happy.
 */
function mustGet<K, V>(map: Map<K, V>, key: K, label: string): V {
  const val = map.get(key);
  if (val === undefined) {
    throw new Error(`Internal error: ${label} "${String(key)}" not found in map.`);
  }
  return val;
}

interface ValidationNode {
  id: string;
  parentId: string | null;
  children: string[];
  depth: number;
}

/**
 * Build adjacency representation and run structural validations:
 *   - Every node has a unique id.
 *   - Exactly one root (parentId === null).
 *   - No cycles.
 *   - No orphans (every non-root has a valid parentId).
 *   - At least one bottom-level node (leaf, no children).
 *
 * @returns A Map of node id → ValidationNode.
 * @throws {Error} if any structural invariant is violated.
 */
function validateAndBuildAdjacency(nodes: readonly HierarchyNode[]): Map<string, ValidationNode> {
  if (nodes.length === 0) {
    throw new Error('Hierarchy must contain at least one node.');
  }

  const map = new Map<string, ValidationNode>();
  const ids = new Set<string>();

  // First pass: collect ids & check uniqueness
  for (const n of nodes) {
    if (ids.has(n.id)) {
      throw new Error(`Duplicate node id: "${n.id}"`);
    }
    ids.add(n.id);
    map.set(n.id, { id: n.id, parentId: n.parentId, children: [], depth: -1 });
  }

  // Second pass: validate parent links & build children lists
  const roots: string[] = [];
  for (const n of nodes) {
    if (n.parentId === null) {
      roots.push(n.id);
    } else {
      if (!ids.has(n.parentId)) {
        throw new Error(
          `Node "${n.id}" references unknown parent "${n.parentId}". ` +
            `All parent ids must exist in the hierarchy.`,
        );
      }
      const parent = mustGet(map, n.parentId, 'parent node');
      parent.children.push(n.id);
    }
  }

  if (roots.length === 0) {
    throw new Error('No root node found — at least one node must have parentId=null.');
  }
  if (roots.length > 1) {
    throw new Error(
      `Multiple root nodes found: ${roots.join(', ')}. A hierarchy must have exactly one root (parentId=null).`,
    );
  }

  // Cycle detection via BFS from root
  const visited = new Set<string>();
  const queue = [roots[0]];

  while (queue.length > 0) {
    const id = queue.shift() as string;
    if (visited.has(id)) {
      throw new Error(`Cycle detected in hierarchy — node "${id}" reached via multiple paths.`);
    }
    visited.add(id);
    const vn = mustGet(map, id, 'node');
    for (const child of vn.children) {
      queue.push(child);
    }
  }

  if (visited.size !== nodes.length) {
    const unreachable = [...ids].filter((id) => !visited.has(id));
    throw new Error(
      `Unreachable nodes detected: ${unreachable.join(', ')}. ` +
        `All nodes must be descendants of the root.`,
    );
  }

  // Compute depths via BFS (topological order)
  const root = roots[0];
  mustGet(map, root, 'root node').depth = 0;
  const bfsQueue = [root];
  while (bfsQueue.length > 0) {
    const id = bfsQueue.shift() as string;
    const vn = mustGet(map, id, 'node');
    for (const child of vn.children) {
      mustGet(map, child, 'child node').depth = vn.depth + 1;
      bfsQueue.push(child);
    }
  }

  // Check for at least one leaf
  const leaves = [...map.values()].filter((vn) => vn.children.length === 0);
  if (leaves.length === 0) {
    throw new Error('Hierarchy must have at least one bottom-level node (leaf).');
  }

  return map;
}

// ---------------------------------------------------------------------------
// Build summing matrix
// ---------------------------------------------------------------------------

/**
 * Build the summing matrix S for the hierarchy.
 *
 * S is an m × n matrix where:
 *   - m = total node count
 *   - n = number of bottom-level (leaf) nodes
 *   - S[i][j] = 1 iff bottom node j is a descendant of (or equal to) node i
 *
 * Row order: topological (parents before children), depth-first.
 * Column order: bottom nodes sorted by id for determinism.
 *
 * @throws {Error} if the hierarchy is not a valid tree (cycles, multiple
 *                 roots, orphans, or no bottom level).
 */
export function buildSummingMatrix(hierarchy: HierarchyDefinition): SummingMatrixResult {
  const adj = validateAndBuildAdjacency(hierarchy.nodes);

  // Identify bottom nodes (leaves) — sorted by id for determinism
  const bottomNodeIds = [...adj.values()]
    .filter((vn) => vn.children.length === 0)
    .map((vn) => vn.id)
    .sort();

  const n = bottomNodeIds.length;

  // All nodes in topological order: BFS by depth, then sort by id within depth
  const allNodes = [...adj.values()].sort((a, b) => {
    if (a.depth !== b.depth) return a.depth - b.depth;
    return a.id.localeCompare(b.id);
  });
  const allNodeIds = allNodes.map((vn) => vn.id);
  const m = allNodeIds.length;

  // Build descendant sets for each node via post-order traversal
  // descendantSet[nodeId] = set of bottom nodes that are descendants
  const descendantSet = new Map<string, Set<string>>();

  function collectDescendants(id: string): Set<string> {
    if (descendantSet.has(id)) return mustGet(descendantSet, id, 'descendant');
    const vn = mustGet(adj, id, 'node');
    const set = new Set<string>();

    if (vn.children.length === 0) {
      // Leaf: its own descendant
      set.add(id);
    } else {
      for (const child of vn.children) {
        const childSet = collectDescendants(child);
        for (const d of childSet) set.add(d);
      }
    }

    descendantSet.set(id, set);
    return set;
  }

  // Start traversal from root — validated earlier so at least one value exists
  const first = adj.values().next();
  if (first.done) throw new Error('Internal error: adjacency map is empty after validation.');
  collectDescendants(first.value.id);

  // Build S matrix
  const S: number[][] = [];
  for (let i = 0; i < m; i++) {
    const row: number[] = [];
    const nodeId = allNodeIds[i];
    const desc = mustGet(descendantSet, nodeId, 'descendant');

    for (let j = 0; j < n; j++) {
      row.push(desc.has(bottomNodeIds[j]) ? 1 : 0);
    }

    S.push(row);
  }

  return { S, allNodeIds, bottomNodeIds };
}
