import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Network,
  Search,
  Sparkles,
  Share2,
  Download,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { LoadingScreen } from '@/components/LoadingScreen'
import { QueryInput } from '@/components/QueryInput'
import { KnowledgeGraph } from '@/components/KnowledgeGraph'
import { SearchResults } from '@/components/SearchResults'

const MOCK_RESULTS = [
  {
    id: '1',
    title: 'Introduction to Machine Learning',
    content: 'Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.',
    source: 'ML Fundamentals.pdf',
    relevance: 0.95,
    type: 'document' as const,
  },
  {
    id: '2',
    title: 'Neural Network Architecture',
    content: 'Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information using a connectionist approach to computation.',
    source: 'Deep Learning Guide.pdf',
    relevance: 0.88,
    type: 'chunk' as const,
  },
  {
    id: '3',
    title: 'Transformer Model',
    content: 'The Transformer architecture, introduced in "Attention Is All You Need", has revolutionized natural language processing by using self-attention mechanisms to process sequential data.',
    source: 'NLP Research Paper',
    relevance: 0.82,
    type: 'entity' as const,
  },
  {
    id: '4',
    title: 'BERT: Pre-training of Deep Bidirectional Transformers',
    content: 'BERT (Bidirectional Encoder Representations from Transformers) is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context.',
    source: 'Google Research',
    relevance: 0.78,
    type: 'document' as const,
  },
  {
    id: '5',
    title: 'Convolutional Neural Networks',
    content: 'CNNs are specialized neural networks for processing data with grid-like topology, such as images. They use convolutional layers to automatically learn hierarchical patterns.',
    source: 'Computer Vision Basics.pdf',
    relevance: 0.72,
    type: 'chunk' as const,
  },
]

export function DashboardPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  const [currentQuery, setCurrentQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'results' | 'graph'>('results')

  const handleSearch = useCallback(async (query: string) => {
    setIsLoading(true)
    setCurrentQuery(query)
    setHasSearched(true)
    await new Promise(resolve => setTimeout(resolve, 2500))
    setIsLoading(false)
  }, [])

  const handleNodeClick = useCallback((node: any) => {
    console.log('Node clicked:', node)
  }, [])

  const handleResultClick = useCallback((result: any) => {
    console.log('Result clicked:', result)
  }, [])

  return (
    <>
      <AnimatePresence>
        {isLoading && <LoadingScreen />}
      </AnimatePresence>

      {!hasSearched ? (
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex flex-col items-center justify-center min-h-[calc(100vh-8rem)]"
        >
          {/* Hero */}
          <div className="text-center mb-12">
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="relative inline-block mb-6"
            >
              <div className="w-24 h-24 mx-auto relative">
                <motion.div
                  className="absolute inset-0 rounded-full border-2 border-dashed border-mosaic-purple/30"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
                />
                <motion.div
                  className="absolute inset-2 rounded-full border-2 border-dashed border-mosaic-teal/30"
                  animate={{ rotate: -360 }}
                  transition={{ duration: 15, repeat: Infinity, ease: 'linear' }}
                />
                <motion.div
                  className="absolute inset-4 rounded-full border-2 border-dashed border-mosaic-coral/30"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 10, repeat: Infinity, ease: 'linear' }}
                />
                <div className="absolute inset-6 rounded-full bg-gradient-to-br from-mosaic-purple via-mosaic-teal to-mosaic-coral flex items-center justify-center">
                  <Sparkles className="w-8 h-8 text-white" />
                </div>
              </div>
            </motion.div>

            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
              className="text-4xl md:text-5xl font-bold mb-4"
            >
              <span className="bg-gradient-to-r from-mosaic-purple via-mosaic-teal to-mosaic-coral bg-clip-text text-transparent">
                Graph-Based RAG
              </span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="text-lg text-muted-foreground max-w-2xl mx-auto"
            >
              Explore your knowledge base through an interactive graph.
              Discover connections, find insights, and get intelligent answers.
            </motion.p>
          </div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.5 }}
            className="w-full max-w-2xl"
          >
            <QueryInput onSubmit={handleSearch} />
          </motion.div>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 }}
            className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-16 w-full max-w-3xl"
          >
            {[
              {
                icon: Network,
                title: 'Visual Graph',
                description: 'Explore connections between concepts and entities',
                color: 'from-mosaic-purple/20 to-mosaic-purple/5',
              },
              {
                icon: Search,
                title: 'Smart Search',
                description: 'Find relevant information with semantic search',
                color: 'from-mosaic-teal/20 to-mosaic-teal/5',
              },
              {
                icon: Sparkles,
                title: 'AI Powered',
                description: 'Get intelligent answers from your knowledge base',
                color: 'from-mosaic-coral/20 to-mosaic-coral/5',
              },
            ].map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.7 + index * 0.1 }}
                className={`
                  p-6 rounded-xl border border-border/50
                  bg-gradient-to-br ${feature.color}
                  hover:border-mosaic-purple/30 transition-colors
                `}
              >
                <feature.icon className="w-8 h-8 text-mosaic-purple mb-3" />
                <h3 className="font-semibold text-foreground mb-1">{feature.title}</h3>
                <p className="text-sm text-muted-foreground">{feature.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="space-y-6"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <QueryInput
              onSubmit={handleSearch}
              isLoading={isLoading}
              placeholder="Ask a follow-up question..."
            />
          </div>

          <div className="flex items-center gap-2 border-b border-border/50">
            <button
              onClick={() => setActiveTab('results')}
              className={`
                px-4 py-2 text-sm font-medium border-b-2 transition-colors
                ${activeTab === 'results'
                  ? 'border-mosaic-purple text-mosaic-purple'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
                }
              `}
            >
              Search Results
            </button>
            <button
              onClick={() => setActiveTab('graph')}
              className={`
                px-4 py-2 text-sm font-medium border-b-2 transition-colors
                ${activeTab === 'graph'
                  ? 'border-mosaic-purple text-mosaic-purple'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
                }
              `}
            >
              Reasoning Graph
            </button>
          </div>

          <AnimatePresence mode="wait">
            {activeTab === 'results' ? (
              <motion.div
                key="results"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                className="space-y-6"
              >
                <div className="glass rounded-xl p-6 border border-mosaic-purple/20">
                  <div className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-mosaic-purple to-mosaic-teal flex items-center justify-center flex-shrink-0">
                      <Sparkles className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-foreground mb-2">AI Summary</h3>
                      <p className="text-sm text-muted-foreground leading-relaxed">
                        Based on your search for &quot;{currentQuery}&quot;, I found {MOCK_RESULTS.length} relevant results.
                        The reasoning graph shows the retrieval path through semantic search, document chunks,
                        and concept extraction to generate a comprehensive answer.
                      </p>
                    </div>
                  </div>
                </div>

                <SearchResults
                  results={MOCK_RESULTS}
                  onResultClick={handleResultClick}
                />
              </motion.div>
            ) : (
              <motion.div
                key="graph"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
              >
                <KnowledgeGraph
                  query={currentQuery}
                  onNodeClick={handleNodeClick}
                  height={600}
                />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="flex items-center justify-end gap-2 pt-4">
            <Button variant="outline" size="sm" className="gap-2">
              <Share2 className="w-4 h-4" />
              Share
            </Button>
            <Button variant="outline" size="sm" className="gap-2">
              <Download className="w-4 h-4" />
              Export
            </Button>
          </div>
        </motion.div>
      )}
    </>
  )
}
