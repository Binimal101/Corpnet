import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Node {
  id: number;
  x: number;
  y: number;
  targetX: number;
  targetY: number;
  radius: number;
  color: string;
  layer: number;
  indexInLayer: number;
  opacity: number;
  scale: number;
  pulsePhase: number;
  built: boolean;
  buildDelay: number;
}

interface Edge {
  from: number;
  to: number;
  opacity: number;
  progress: number;
  built: boolean;
  buildDelay: number;
}

const COLORS = [
  '#a855f7', // purple
  '#14b8a6', // teal
  '#f97316', // coral
  '#eab308', // gold
  '#3b82f6', // blue
  '#ec4899', // rose
];

// Neural network layer configuration
const LAYERS = [4, 6, 8, 6, 4]; // Nodes per layer

export function LoadingScreen() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nodesRef = useRef<Node[]>([]);
  const edgesRef = useRef<Edge[]>([]);
  const animationRef = useRef<number | undefined>(undefined);
  const [buildPhase, setBuildPhase] = useState<'nodes' | 'edges' | 'complete'>('nodes');
  const [buildProgress, setBuildProgress] = useState(0);

  const initNeuralNetwork = useCallback((width: number, height: number) => {
    const nodes: Node[] = [];
    const edges: Edge[] = [];
    
    const layerSpacing = width * 0.7 / (LAYERS.length - 1);
    const startX = width * 0.15;
    const centerY = height / 2;
    
    let nodeId = 0;
    const layerNodes: number[][] = [];
    
    // Create nodes in layers
    LAYERS.forEach((nodeCount, layerIndex) => {
      const layerX = startX + layerIndex * layerSpacing;
      const layerNodeIds: number[] = [];
      const verticalSpacing = Math.min(80, height * 0.6 / (nodeCount + 1));
      const layerStartY = centerY - (nodeCount - 1) * verticalSpacing / 2;
      
      for (let i = 0; i < nodeCount; i++) {
        const targetY = layerStartY + i * verticalSpacing;
        
        nodes.push({
          id: nodeId,
          x: layerX,
          y: targetY + (Math.random() - 0.5) * 100, // Start slightly offset
          targetX: layerX,
          targetY: targetY,
          radius: layerIndex === 0 || layerIndex === LAYERS.length - 1 ? 8 : 6,
          color: COLORS[layerIndex % COLORS.length],
          layer: layerIndex,
          indexInLayer: i,
          opacity: 0,
          scale: 0,
          pulsePhase: Math.random() * Math.PI * 2,
          built: false,
          buildDelay: layerIndex * 300 + i * 50, // Staggered build
        });
        
        layerNodeIds.push(nodeId);
        nodeId++;
      }
      
      layerNodes.push(layerNodeIds);
    });
    
    // Create edges between layers
    for (let layerIndex = 0; layerIndex < LAYERS.length - 1; layerIndex++) {
      const currentLayer = layerNodes[layerIndex];
      const nextLayer = layerNodes[layerIndex + 1];
      
      currentLayer.forEach((fromId, fromIndex) => {
        nextLayer.forEach((toId, toIndex) => {
          // Create dense connections but not all-to-all for visual clarity
          const shouldConnect = 
            Math.abs(fromIndex - toIndex) <= 2 || 
            Math.random() < 0.4;
          
          if (shouldConnect) {
            edges.push({
              from: fromId,
              to: toId,
              opacity: 0,
              progress: 0,
              built: false,
              buildDelay: nodes[fromId].buildDelay + 150 + Math.random() * 100,
            });
          }
        });
      });
    }
    
    nodesRef.current = nodes;
    edgesRef.current = edges;
  }, []);

  const animate = useCallback((timestamp: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const width = canvas.width / (window.devicePixelRatio || 1);
    const height = canvas.height / (window.devicePixelRatio || 1);
    const nodes = nodesRef.current;
    const edges = edgesRef.current;

    ctx.clearRect(0, 0, width, height);

    let nodesBuilt = 0;
    let edgesBuilt = 0;

    // Update and draw nodes
    nodes.forEach((node) => {
      // Build animation
      if (!node.built) {
        if (timestamp > node.buildDelay) {
          node.opacity = Math.min(1, node.opacity + 0.08);
          node.scale = Math.min(1, node.scale + 0.08);
          
          // Spring to target position
          const dx = node.targetX - node.x;
          const dy = node.targetY - node.y;
          node.x += dx * 0.1;
          node.y += dy * 0.1;
          
          if (node.opacity >= 1 && node.scale >= 1 && Math.abs(dx) < 0.5 && Math.abs(dy) < 0.5) {
            node.built = true;
          }
        }
      } else {
        nodesBuilt++;
        // Idle animation
        node.pulsePhase += 0.03;
        
        // Subtle floating
        node.y = node.targetY + Math.sin(node.pulsePhase) * 3;
      }
      
      if (node.opacity > 0) {
        const pulseScale = node.built ? 1 + Math.sin(node.pulsePhase) * 0.1 : 1;
        const currentRadius = node.radius * node.scale * pulseScale;
        
        // Draw glow
        const glowGradient = ctx.createRadialGradient(
          node.x, node.y, 0,
          node.x, node.y, currentRadius * 4
        );
        glowGradient.addColorStop(0, node.color + Math.floor(node.opacity * 40).toString(16).padStart(2, '0'));
        glowGradient.addColorStop(0.3, node.color + Math.floor(node.opacity * 20).toString(16).padStart(2, '0'));
        glowGradient.addColorStop(1, 'transparent');
        
        ctx.beginPath();
        ctx.arc(node.x, node.y, currentRadius * 4, 0, Math.PI * 2);
        ctx.fillStyle = glowGradient;
        ctx.fill();
        
        // Draw node core
        ctx.beginPath();
        ctx.arc(node.x, node.y, currentRadius, 0, Math.PI * 2);
        ctx.fillStyle = node.color;
        ctx.globalAlpha = node.opacity;
        ctx.fill();
        ctx.globalAlpha = 1;
        
        // Draw inner highlight
        ctx.beginPath();
        ctx.arc(node.x - currentRadius * 0.3, node.y - currentRadius * 0.3, currentRadius * 0.3, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
        ctx.globalAlpha = node.opacity;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    });

    // Update and draw edges
    edges.forEach((edge) => {
      const fromNode = nodes[edge.from];
      const toNode = nodes[edge.to];
      
      if (!edge.built) {
        if (timestamp > edge.buildDelay && fromNode.opacity > 0.5 && toNode.opacity > 0.5) {
          edge.progress = Math.min(1, edge.progress + 0.05);
          edge.opacity = Math.min(0.6, edge.opacity + 0.03);
          
          if (edge.progress >= 1) {
            edge.built = true;
          }
        }
      } else {
        edgesBuilt++;
        edge.opacity = 0.4 + Math.sin(timestamp * 0.002) * 0.1;
      }
      
      if (edge.progress > 0) {
        const gradient = ctx.createLinearGradient(fromNode.x, fromNode.y, toNode.x, toNode.y);
        gradient.addColorStop(0, fromNode.color + '40');
        gradient.addColorStop(1, toNode.color + '40');
        
        ctx.beginPath();
        ctx.moveTo(fromNode.x, fromNode.y);
        
        // Draw partial line based on progress
        const dx = toNode.x - fromNode.x;
        const dy = toNode.y - fromNode.y;
        const endX = fromNode.x + dx * edge.progress;
        const endY = fromNode.y + dy * edge.progress;
        
        ctx.lineTo(endX, endY);
        ctx.strokeStyle = gradient;
        ctx.lineWidth = 1.5;
        ctx.globalAlpha = edge.opacity;
        ctx.stroke();
        ctx.globalAlpha = 1;
        
        // Draw spark at the end of building edge
        if (edge.progress < 1 && edge.progress > 0.1) {
          ctx.beginPath();
          ctx.arc(endX, endY, 3, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.globalAlpha = 0.8;
          ctx.fill();
          ctx.globalAlpha = 1;
        }
      }
    });

    // Update build progress
    const totalNodes = nodes.length;
    const totalEdges = edges.length;
    const totalItems = totalNodes + totalEdges;
    const builtItems = nodesBuilt + edgesBuilt;
    setBuildProgress(Math.round((builtItems / totalItems) * 100));
    
    if (nodesBuilt === totalNodes && buildPhase === 'nodes') {
      setBuildPhase('edges');
    } else if (edgesBuilt === totalEdges && buildPhase === 'edges') {
      setBuildPhase('complete');
    }

    animationRef.current = requestAnimationFrame(animate);
  }, [buildPhase]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const handleResize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = window.innerWidth * dpr;
      canvas.height = window.innerHeight * dpr;
      canvas.style.width = `${window.innerWidth}px`;
      canvas.style.height = `${window.innerHeight}px`;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.scale(dpr, dpr);
      }
      
      initNeuralNetwork(window.innerWidth, window.innerHeight);
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    
    animationRef.current = requestAnimationFrame(animate);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [initNeuralNetwork, animate]);

  return (
    <div className="fixed inset-0 bg-background flex items-center justify-center z-50">
      {/* Background grid */}
      <div className="absolute inset-0 opacity-5">
        <svg width="100%" height="100%">
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="currentColor" strokeWidth="0.5"/>
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </div>
      
      <canvas
        ref={canvasRef}
        className="absolute inset-0"
        style={{ opacity: 0.9 }}
      />
      
      <div className="relative z-10 flex flex-col items-center">
        {/* Central logo with rings */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.6, ease: "easeOut" }}
          className="mb-8"
        >
          <div className="relative w-32 h-32">
            {/* Outer rotating ring */}
            <motion.div
              className="absolute inset-0 rounded-full border-2 border-dashed border-mosaic-purple/50"
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Middle rotating ring */}
            <motion.div
              className="absolute inset-3 rounded-full border-2 border-dashed border-mosaic-teal/50"
              animate={{ rotate: -360 }}
              transition={{ duration: 15, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Inner rotating ring */}
            <motion.div
              className="absolute inset-6 rounded-full border-2 border-dashed border-mosaic-coral/50"
              animate={{ rotate: 360 }}
              transition={{ duration: 10, repeat: Infinity, ease: "linear" }}
            />
            
            {/* Core pulsing circle */}
            <motion.div
              className="absolute inset-9 rounded-full bg-gradient-to-br from-mosaic-purple via-mosaic-teal to-mosaic-coral"
              animate={{ 
                scale: [1, 1.15, 1],
                opacity: [0.8, 1, 0.8]
              }}
              transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            />
            
            {/* Center icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <svg 
                className="w-10 h-10 text-white" 
                fill="none" 
                viewBox="0 0 24 24" 
                stroke="currentColor"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={1.5} 
                  d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" 
                />
              </svg>
            </div>
          </div>
        </motion.div>
        
        {/* Status text */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="text-center"
        >
          <h2 className="text-3xl font-bold bg-gradient-to-r from-mosaic-purple via-mosaic-teal to-mosaic-coral bg-clip-text text-transparent">
            Building Neural Network
          </h2>
          
          <AnimatePresence mode="wait">
            <motion.p
              key={buildPhase}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="mt-3 text-muted-foreground text-sm"
            >
              {buildPhase === 'nodes' && 'Initializing nodes across layers...'}
              {buildPhase === 'edges' && 'Establishing connections...'}
              {buildPhase === 'complete' && 'Network ready. Processing query...'}
            </motion.p>
          </AnimatePresence>
        </motion.div>
        
        {/* Progress bar */}
        <motion.div
          initial={{ opacity: 0, width: 0 }}
          animate={{ opacity: 1, width: 256 }}
          transition={{ duration: 0.5, delay: 0.5 }}
          className="mt-6"
        >
          <div className="w-64 h-2 bg-muted rounded-full overflow-hidden">
            <motion.div
              className="h-full bg-gradient-to-r from-mosaic-purple via-mosaic-teal to-mosaic-coral"
              initial={{ width: 0 }}
              animate={{ width: `${buildProgress}%` }}
              transition={{ duration: 0.3 }}
            />
          </div>
          <p className="text-center text-xs text-muted-foreground mt-2">
            {buildProgress}%
          </p>
        </motion.div>
        
        {/* Layer indicators */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          className="flex items-center gap-3 mt-8"
        >
          {LAYERS.map((count, i) => (
            <motion.div
              key={i}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.7 + i * 0.1 }}
              className="flex flex-col items-center gap-1"
            >
              <div className={`
                w-8 h-8 rounded-lg flex items-center justify-center text-xs font-bold
                ${buildPhase === 'complete' || (buildPhase === 'edges' && i < LAYERS.length - 1) || (buildPhase === 'nodes' && i === 0)
                  ? 'bg-mosaic-purple/30 text-mosaic-purple' 
                  : 'bg-muted text-muted-foreground'}
                transition-colors duration-300
              `}>
                {count}
              </div>
              <span className="text-[10px] text-muted-foreground uppercase">
                {i === 0 ? 'Input' : i === LAYERS.length - 1 ? 'Output' : `L${i}`}
              </span>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </div>
  );
}
