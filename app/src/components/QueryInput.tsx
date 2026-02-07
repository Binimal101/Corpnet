import { useState, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Search, Sparkles, Loader2, Mic, History, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface QueryInputProps {
  onSubmit: (query: string) => void;
  isLoading?: boolean;
  placeholder?: string;
  suggestions?: string[];
}

const EXAMPLE_QUERIES = [
  "What are the main concepts in machine learning?",
  "Explain the relationship between neural networks and deep learning",
  "Find documents about natural language processing",
  "What are the key entities in this knowledge base?",
];

export function QueryInput({ 
  onSubmit, 
  isLoading, 
  placeholder = "Ask anything about your knowledge graph...",
  suggestions = EXAMPLE_QUERIES 
}: QueryInputProps) {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
      setShowSuggestions(false);
    }
  }, [query, isLoading, onSubmit]);

  const handleSuggestionClick = useCallback((suggestion: string) => {
    setQuery(suggestion);
    onSubmit(suggestion);
    setShowSuggestions(false);
  }, [onSubmit]);

  const clearQuery = useCallback(() => {
    setQuery('');
    inputRef.current?.focus();
  }, []);

  return (
    <div className="w-full max-w-3xl mx-auto">
      <motion.form
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="relative"
      >
        {/* Main input container */}
        <div
          className={`
            relative rounded-2xl overflow-hidden
            bg-gradient-to-br from-card/80 to-card/40
            backdrop-blur-xl
            border-2 transition-all duration-300
            ${isFocused 
              ? 'border-mosaic-purple/50 shadow-lg shadow-mosaic-purple/20' 
              : 'border-border/50 hover:border-border'
            }
          `}
        >
          {/* Animated gradient border on focus */}
          <AnimatePresence>
            {isFocused && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="absolute inset-0 rounded-2xl pointer-events-none"
                style={{
                  background: 'linear-gradient(90deg, hsl(265, 85%, 55%), hsl(175, 85%, 45%), hsl(15, 90%, 60%), hsl(45, 95%, 55%))',
                  backgroundSize: '300% 300%',
                  animation: 'gradient-shift 3s ease infinite',
                  opacity: 0.3,
                  filter: 'blur(8px)',
                  zIndex: -1,
                }}
              />
            )}
          </AnimatePresence>

          <div className="flex items-center gap-3 p-4">
            {/* Search icon */}
            <div className={`
              flex-shrink-0 w-10 h-10 rounded-xl
              flex items-center justify-center
              transition-all duration-300
              ${isFocused 
                ? 'bg-gradient-to-br from-mosaic-purple to-mosaic-teal text-white' 
                : 'bg-muted text-muted-foreground'
              }
            `}>
              {isLoading ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Search className="w-5 h-5" />
              )}
            </div>

            {/* Input field */}
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onFocus={() => {
                setIsFocused(true);
                setShowSuggestions(true);
              }}
              onBlur={() => {
                setIsFocused(false);
                setTimeout(() => setShowSuggestions(false), 200);
              }}
              placeholder={placeholder}
              disabled={isLoading}
              className="
                flex-1 bg-transparent border-none outline-none
                text-foreground placeholder:text-muted-foreground
                text-base md:text-lg
                disabled:cursor-not-allowed
              "
            />

            {/* Action buttons */}
            <div className="flex items-center gap-2">
              {query && (
                <motion.button
                  type="button"
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  onClick={clearQuery}
                  className="p-2 rounded-lg hover:bg-muted text-muted-foreground transition-colors"
                >
                  <X className="w-4 h-4" />
                </motion.button>
              )}

              <Button
                type="button"
                variant="ghost"
                size="icon"
                className="rounded-lg text-muted-foreground hover:text-foreground"
              >
                <Mic className="w-4 h-4" />
              </Button>

              <Button
                type="submit"
                disabled={!query.trim() || isLoading}
                className={`
                  rounded-xl px-6 py-2
                  bg-gradient-to-r from-mosaic-purple to-mosaic-teal
                  text-white font-medium
                  hover:opacity-90 transition-opacity
                  disabled:opacity-50 disabled:cursor-not-allowed
                  flex items-center gap-2
                `}
              >
                <Sparkles className="w-4 h-4" />
                <span className="hidden sm:inline">Search</span>
              </Button>
            </div>
          </div>
        </div>

        {/* Suggestions dropdown */}
        <AnimatePresence>
          {showSuggestions && !query && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.2 }}
              className="absolute top-full left-0 right-0 mt-2 z-50"
            >
              <div className="glass rounded-xl p-4 border border-border/50">
                <div className="flex items-center gap-2 text-xs text-muted-foreground mb-3">
                  <History className="w-3 h-3" />
                  <span>Try these examples</span>
                </div>
                <div className="space-y-1">
                  {suggestions.map((suggestion, index) => (
                    <motion.button
                      key={index}
                      type="button"
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="
                        w-full text-left px-3 py-2 rounded-lg
                        text-sm text-foreground
                        hover:bg-mosaic-purple/10 hover:text-mosaic-purple
                        transition-colors
                        flex items-center gap-2
                      "
                    >
                      <Search className="w-3 h-3 text-muted-foreground" />
                      {suggestion}
                    </motion.button>
                  ))}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </motion.form>

      {/* Quick action chips */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="flex flex-wrap justify-center gap-2 mt-4"
      >
        {['Concepts', 'Entities', 'Documents', 'Topics'].map((chip, index) => (
          <motion.button
            key={chip}
            type="button"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: 0.4 + index * 0.05 }}
            onClick={() => handleSuggestionClick(`Show all ${chip.toLowerCase()}`)}
            className="
              px-3 py-1.5 rounded-full text-xs
              bg-muted/50 text-muted-foreground
              hover:bg-mosaic-purple/20 hover:text-mosaic-purple
              transition-colors
              border border-border/50
            "
          >
            {chip}
          </motion.button>
        ))}
      </motion.div>
    </div>
  );
}
