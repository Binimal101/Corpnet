import { useRef, useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import * as d3 from 'd3';

interface TreeNode {
  id: string;
  name: string;
  type: 'query' | 'retrieval' | 'chunk' | 'concept' | 'reasoning' | 'answer';
  description?: string;
  relevance?: number;
  children?: TreeNode[];
  _children?: TreeNode[];
  x?: number;
  y?: number;
  depth?: number;
}

interface HierarchyNode extends d3.HierarchyNode<TreeNode> {
  x: number;
  y: number;
}

interface KnowledgeGraphProps {
  query: string;
  onNodeClick?: (node: TreeNode) => void;
  height?: number;
}

const NODE_COLORS: Record<string, string> = {
  query: '#a855f7',      // purple
  retrieval: '#14b8a6',  // teal
  chunk: '#3b82f6',      // blue
  concept: '#f97316',    // coral
  reasoning: '#eab308',  // gold
  answer: '#22c55e',     // green
};

const NODE_ICONS: Record<string, string> = {
  query: '?',
  retrieval: '🔍',
  chunk: '📄',
  concept: '💡',
  reasoning: '⚡',
  answer: '✓',
};

// Generate RAG reasoning tree based on query
function generateRAGTree(query: string): TreeNode {
  return {
    id: 'root',
    name: query,
    type: 'query',
    children: [
      {
        id: 'retrieval-1',
        name: 'Semantic Search',
        type: 'retrieval',
        description: 'Searching vector database for relevant documents',
        children: [
          {
            id: 'chunk-1',
            name: 'ML Fundamentals',
            type: 'chunk',
            description: 'Introduction to machine learning concepts',
            relevance: 0.95,
            children: [
              {
                id: 'concept-1',
                name: 'Supervised Learning',
                type: 'concept',
                description: 'Learning with labeled training data',
              },
              {
                id: 'concept-2',
                name: 'Neural Networks',
                type: 'concept',
                description: 'Computing systems inspired by biological neurons',
              },
            ],
          },
          {
            id: 'chunk-2',
            name: 'Deep Learning Guide',
            type: 'chunk',
            description: 'Advanced neural network architectures',
            relevance: 0.88,
            children: [
              {
                id: 'concept-3',
                name: 'Backpropagation',
                type: 'concept',
                description: 'Algorithm for training neural networks',
              },
              {
                id: 'concept-4',
                name: 'CNN Architecture',
                type: 'concept',
                description: 'Convolutional layers for image processing',
              },
            ],
          },
          {
            id: 'chunk-3',
            name: 'Transformer Paper',
            type: 'chunk',
            description: 'Attention is All You Need',
            relevance: 0.82,
            children: [
              {
                id: 'concept-5',
                name: 'Self-Attention',
                type: 'concept',
                description: 'Mechanism for capturing global dependencies',
              },
              {
                id: 'concept-6',
                name: 'Multi-Head Attention',
                type: 'concept',
                description: 'Parallel attention mechanisms',
              },
            ],
          },
        ],
      },
      {
        id: 'reasoning-1',
        name: 'Graph Reasoning',
        type: 'reasoning',
        description: 'Connecting concepts through knowledge graph',
        children: [
          {
            id: 'concept-7',
            name: 'Feature Extraction',
            type: 'concept',
            description: 'Learning hierarchical representations',
          },
          {
            id: 'concept-8',
            name: 'Pattern Recognition',
            type: 'concept',
            description: 'Identifying structures in data',
          },
        ],
      },
      {
        id: 'reasoning-2',
        name: 'Synthesis',
        type: 'reasoning',
        description: 'Combining information from multiple sources',
        children: [
          {
            id: 'answer-1',
            name: 'Generated Response',
            type: 'answer',
            description: 'Comprehensive answer based on retrieved context',
          },
        ],
      },
    ],
  };
}

export function KnowledgeGraph({ query, onNodeClick, height = 600 }: KnowledgeGraphProps) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<TreeNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<TreeNode | null>(null);
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set(['root', 'retrieval-1']));

  const treeData = useMemo(() => generateRAGTree(query), [query]);

  // Prepare data with collapsed/expanded state
  const processedData = useMemo(() => {
    const processNode = (node: TreeNode): TreeNode => {
      const isExpanded = expandedNodes.has(node.id);
      const processed: TreeNode = { ...node };
      
      if (node.children) {
        if (isExpanded) {
          processed.children = node.children.map(processNode);
        } else {
          processed._children = node.children.map(processNode);
          processed.children = undefined;
        }
      }
      
      return processed;
    };
    
    return processNode(treeData);
  }, [treeData, expandedNodes]);

  useEffect(() => {
    if (!svgRef.current) return;

    const svg = d3.select(svgRef.current);
    const width = svgRef.current.clientWidth;
    const margin = { top: 40, right: 120, bottom: 40, left: 120 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    svg.selectAll('*').remove();

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create tree layout
    const treeLayout = d3.tree<TreeNode>()
      .size([innerHeight, innerWidth])
      .separation((a: d3.HierarchyNode<TreeNode>, b: d3.HierarchyNode<TreeNode>) => 
        (a.parent === b.parent ? 1.5 : 2) / ((a.depth || 1))
      );

    const root = d3.hierarchy(processedData) as HierarchyNode;
    treeLayout(root);

    // Draw links with gradient
    const links = g.selectAll<SVGPathElement, d3.HierarchyLink<TreeNode>>('.link')
      .data(root.links())
      .enter()
      .append('path')
      .attr('class', 'link')
      .attr('d', d3.linkHorizontal<d3.HierarchyLink<TreeNode>, HierarchyNode>()
        .x((d: HierarchyNode) => d.y)
        .y((d: HierarchyNode) => d.x)
      )
      .attr('fill', 'none')
      .attr('stroke', (d: d3.HierarchyLink<TreeNode>) => {
        const gradientId = `gradient-${d.source.data.id}-${d.target.data.id}`;
        const gradient = svg.append('defs')
          .append('linearGradient')
          .attr('id', gradientId)
          .attr('gradientUnits', 'userSpaceOnUse')
          .attr('x1', (d.source as HierarchyNode).y)
          .attr('y1', (d.source as HierarchyNode).x)
          .attr('x2', (d.target as HierarchyNode).y)
          .attr('y2', (d.target as HierarchyNode).x);
        
        gradient.append('stop')
          .attr('offset', '0%')
          .attr('stop-color', NODE_COLORS[d.source.data.type]);
        
        gradient.append('stop')
          .attr('offset', '100%')
          .attr('stop-color', NODE_COLORS[d.target.data.type]);
        
        return `url(#${gradientId})`;
      })
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.6)
      .attr('stroke-linecap', 'round');

    // Animate links
    links.attr('stroke-dasharray', function(this: SVGPathElement) {
      const length = this.getTotalLength();
      return `${length} ${length}`;
    })
    .attr('stroke-dashoffset', function(this: SVGPathElement) {
      return this.getTotalLength();
    })
    .transition()
    .duration(800)
    .delay((_: d3.HierarchyLink<TreeNode>, i: number) => i * 100)
    .attr('stroke-dashoffset', 0);

    // Draw nodes
    const nodes = g.selectAll<SVGGElement, HierarchyNode>('.node')
      .data(root.descendants())
      .enter()
      .append('g')
      .attr('class', 'node')
      .attr('transform', (d: HierarchyNode) => `translate(${d.y},${d.x})`)
      .style('cursor', 'pointer')
      .on('click', (event: MouseEvent, d: HierarchyNode) => {
        event.stopPropagation();
        setSelectedNode(d.data);
        onNodeClick?.(d.data);
        
        // Toggle expansion
        if (d.data.children || d.data._children) {
          setExpandedNodes(prev => {
            const next = new Set(prev);
            if (next.has(d.data.id)) {
              next.delete(d.data.id);
            } else {
              next.add(d.data.id);
            }
            return next;
          });
        }
      })
      .on('mouseenter', (_: MouseEvent, d: HierarchyNode) => setHoveredNode(d.data))
      .on('mouseleave', () => setHoveredNode(null));

    // Node circles with glow
    nodes.append('circle')
      .attr('r', (d: HierarchyNode) => d.data.type === 'query' ? 25 : d.data.type === 'answer' ? 22 : 18)
      .attr('fill', (d: HierarchyNode) => NODE_COLORS[d.data.type])
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .attr('stroke-opacity', 0.3)
      .style('filter', (d: HierarchyNode) => `drop-shadow(0 0 ${d.data.type === 'query' ? 15 : 10}px ${NODE_COLORS[d.data.type]})`);

    // Inner highlight
    nodes.append('circle')
      .attr('r', (d: HierarchyNode) => (d.data.type === 'query' ? 25 : d.data.type === 'answer' ? 22 : 18) * 0.4)
      .attr('fill', 'rgba(255,255,255,0.3)')
      .attr('cx', -5)
      .attr('cy', -5);

    // Icons/labels
    nodes.append('text')
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle')
      .text((d: HierarchyNode) => NODE_ICONS[d.data.type])
      .style('font-size', (d: HierarchyNode) => d.data.type === 'query' ? '16px' : '12px')
      .style('pointer-events', 'none');

    // Node labels
    nodes.append('text')
      .attr('dy', (d: HierarchyNode) => d.data.type === 'query' ? 40 : 32)
      .attr('text-anchor', 'middle')
      .text((d: HierarchyNode) => d.data.name)
      .style('font-size', '11px')
      .style('font-weight', '600')
      .style('fill', '#e5e7eb')
      .style('pointer-events', 'none');

    // Relevance score for chunks
    nodes.filter((d: HierarchyNode) => d.data.type === 'chunk' && !!d.data.relevance)
      .append('text')
      .attr('dy', 42)
      .attr('text-anchor', 'middle')
      .text((d: HierarchyNode) => `${Math.round((d.data.relevance || 0) * 100)}%`)
      .style('font-size', '9px')
      .style('fill', '#22c55e')
      .style('pointer-events', 'none');

    // Expand/collapse indicator
    nodes.filter((d: HierarchyNode) => !!(d.data.children || d.data._children))
      .append('circle')
      .attr('r', 6)
      .attr('cx', (d: HierarchyNode) => (d.data.type === 'query' ? 25 : d.data.type === 'answer' ? 22 : 18) + 8)
      .attr('fill', '#374151')
      .attr('stroke', '#6b7280')
      .attr('stroke-width', 1);

    nodes.filter((d: HierarchyNode) => !!(d.data.children || d.data._children))
      .append('text')
      .attr('x', (d: HierarchyNode) => (d.data.type === 'query' ? 25 : d.data.type === 'answer' ? 22 : 18) + 8)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'middle')
      .text((d: HierarchyNode) => d.data.children ? '−' : '+')
      .style('font-size', '10px')
      .style('fill', '#fff')
      .style('pointer-events', 'none');

    // Animate nodes
    nodes.attr('opacity', 0)
      .transition()
      .duration(500)
      .delay((_: HierarchyNode, i: number) => i * 80)
      .attr('opacity', 1);

  }, [processedData, height, onNodeClick]);

  const displayNode = hoveredNode || selectedNode;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="relative w-full rounded-xl overflow-hidden border border-border/50 bg-card/30"
      style={{ height }}
    >
      {/* Header */}
      <div className="absolute top-0 left-0 right-0 z-10 p-4 flex items-center justify-between bg-gradient-to-b from-background/90 to-transparent">
        <div className="flex items-center gap-3">
          <div className="w-3 h-3 rounded-full bg-mosaic-purple animate-pulse" />
          <span className="text-sm font-medium text-foreground">RAG Reasoning Path</span>
        </div>
        <div className="flex items-center gap-4 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: NODE_COLORS.retrieval }} />
            Retrieval
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: NODE_COLORS.chunk }} />
            Context
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: NODE_COLORS.concept }} />
            Concepts
          </span>
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full" style={{ background: NODE_COLORS.reasoning }} />
            Reasoning
          </span>
        </div>
      </div>

      {/* SVG Graph */}
      <svg
        ref={svgRef}
        className="w-full h-full"
        style={{ minHeight: height }}
      />

      {/* Node info panel */}
      <AnimatePresence>
        {displayNode && (
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 20 }}
            className="absolute top-16 right-4 z-20 glass rounded-lg p-4 max-w-xs border border-border/50"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="text-lg">{NODE_ICONS[displayNode.type]}</span>
              <span className="font-semibold text-foreground">{displayNode.name}</span>
            </div>
            <div className="text-xs text-mosaic-purple uppercase tracking-wider mb-2">
              {displayNode.type}
            </div>
            {displayNode.description && (
              <p className="text-sm text-muted-foreground">
                {displayNode.description}
              </p>
            )}
            {displayNode.relevance && (
              <div className="mt-3 flex items-center gap-2">
                <span className="text-xs text-muted-foreground">Relevance:</span>
                <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden">
                  <div 
                    className="h-full bg-mosaic-emerald rounded-full"
                    style={{ width: `${displayNode.relevance * 100}%` }}
                  />
                </div>
                <span className="text-xs text-mosaic-emerald">
                  {Math.round(displayNode.relevance * 100)}%
                </span>
              </div>
            )}
            <p className="mt-3 text-xs text-muted-foreground">
              Click to {(displayNode.children || displayNode._children) ? 'toggle' : 'select'}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 glass rounded-lg p-3">
        <div className="text-xs font-medium text-muted-foreground mb-2">Node Types</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1">
          {Object.entries(NODE_COLORS).map(([type, color]) => (
            <div key={type} className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-xs text-foreground capitalize">{type}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
