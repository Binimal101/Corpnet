import { motion, AnimatePresence } from 'framer-motion';
import { FileText, Link2, Hash, Sparkles, ChevronRight } from 'lucide-react';

interface SearchResult {
  id: string;
  title: string;
  content: string;
  source: string;
  relevance: number;
  type: 'document' | 'chunk' | 'entity';
  metadata?: Record<string, any>;
}

interface SearchResultsProps {
  results: SearchResult[];
  isLoading?: boolean;
  onResultClick?: (result: SearchResult) => void;
}

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1,
    },
  },
};

const itemVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.4,
      ease: [0.25, 0.46, 0.45, 0.96] as const,
    },
  },
};

function getTypeIcon(type: SearchResult['type']) {
  switch (type) {
    case 'document':
      return <FileText className="w-4 h-4" />;
    case 'chunk':
      return <Hash className="w-4 h-4" />;
    case 'entity':
      return <Link2 className="w-4 h-4" />;
    default:
      return <Sparkles className="w-4 h-4" />;
  }
}

function getTypeColor(type: SearchResult['type']) {
  switch (type) {
    case 'document':
      return 'from-mosaic-purple/20 to-mosaic-purple/5 border-mosaic-purple/30';
    case 'chunk':
      return 'from-mosaic-teal/20 to-mosaic-teal/5 border-mosaic-teal/30';
    case 'entity':
      return 'from-mosaic-coral/20 to-mosaic-coral/5 border-mosaic-coral/30';
    default:
      return 'from-mosaic-gold/20 to-mosaic-gold/5 border-mosaic-gold/30';
  }
}

function RelevanceBadge({ score }: { score: number }) {
  const percentage = Math.round(score * 100);
  let colorClass = 'text-mosaic-coral';
  if (percentage >= 80) colorClass = 'text-mosaic-emerald';
  else if (percentage >= 60) colorClass = 'text-mosaic-gold';
  
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 bg-muted rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className={`h-full rounded-full ${
            percentage >= 80 ? 'bg-mosaic-emerald' :
            percentage >= 60 ? 'bg-mosaic-gold' :
            'bg-mosaic-coral'
          }`}
        />
      </div>
      <span className={`text-xs font-medium ${colorClass}`}>{percentage}%</span>
    </div>
  );
}

export function SearchResults({ results, isLoading, onResultClick }: SearchResultsProps) {
  if (isLoading) {
    return (
      <div className="space-y-4">
        {[1, 2, 3].map((i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass rounded-xl p-5 shimmer"
          >
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-lg bg-muted animate-pulse" />
              <div className="flex-1 space-y-3">
                <div className="h-4 bg-muted rounded w-3/4 animate-pulse" />
                <div className="h-3 bg-muted rounded w-full animate-pulse" />
                <div className="h-3 bg-muted rounded w-2/3 animate-pulse" />
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center py-12"
      >
        <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted flex items-center justify-center">
          <Sparkles className="w-8 h-8 text-muted-foreground" />
        </div>
        <h3 className="text-lg font-medium text-foreground">No results found</h3>
        <p className="text-sm text-muted-foreground mt-1">
          Try adjusting your search query or explore the knowledge graph
        </p>
      </motion.div>
    );
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className="space-y-4"
    >
      <AnimatePresence>
        {results.map((result) => (
          <motion.div
            key={result.id}
            variants={itemVariants}
            layout
            whileHover={{ scale: 1.01, y: -2 }}
            whileTap={{ scale: 0.99 }}
            onClick={() => onResultClick?.(result)}
            className={`
              relative group cursor-pointer rounded-xl p-5
              bg-gradient-to-br ${getTypeColor(result.type)}
              border backdrop-blur-sm
              transition-all duration-300
              hover:shadow-lg hover:shadow-mosaic-purple/10
            `}
          >
            {/* Mosaic tile decoration */}
            <div className="absolute top-0 right-0 w-20 h-20 opacity-10 pointer-events-none">
              <div className="absolute top-2 right-2 w-4 h-4 rounded-sm bg-current" />
              <div className="absolute top-2 right-8 w-4 h-4 rounded-sm bg-current" />
              <div className="absolute top-8 right-2 w-4 h-4 rounded-sm bg-current" />
            </div>

            <div className="flex items-start gap-4">
              {/* Icon */}
              <div className={`
                w-10 h-10 rounded-lg flex items-center justify-center
                bg-gradient-to-br ${getTypeColor(result.type)}
                text-foreground
              `}>
                {getTypeIcon(result.type)}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-4">
                  <h3 className="text-base font-semibold text-foreground group-hover:text-mosaic-purple transition-colors line-clamp-1">
                    {result.title}
                  </h3>
                  <RelevanceBadge score={result.relevance} />
                </div>

                <p className="mt-2 text-sm text-muted-foreground line-clamp-2">
                  {result.content}
                </p>

                <div className="mt-3 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs text-muted-foreground">
                    <span className="px-2 py-0.5 rounded-full bg-muted capitalize">
                      {result.type}
                    </span>
                    <span className="truncate max-w-[200px]">{result.source}</span>
                  </div>

                  <motion.div
                    initial={{ x: -5, opacity: 0 }}
                    whileHover={{ x: 0, opacity: 1 }}
                    className="flex items-center gap-1 text-xs text-mosaic-purple opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    View details
                    <ChevronRight className="w-3 h-3" />
                  </motion.div>
                </div>
              </div>
            </div>

            {/* Hover gradient overlay */}
            <div className="absolute inset-0 rounded-xl bg-gradient-to-r from-mosaic-purple/5 via-transparent to-mosaic-teal/5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />
          </motion.div>
        ))}
      </AnimatePresence>
    </motion.div>
  );
}
