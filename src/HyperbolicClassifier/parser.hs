{-# LANGUAGE OverloadedStrings #-}

module Main where

import Data.Void
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as TIO
import Text.Megaparsec
import Text.Megaparsec.Char
import qualified Text.Megaparsec.Char.Lexer as L
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import Data.List (foldl')

type Parser = Parsec Void Text

type SynsetID = Int

data Synset = Synset
  { synsetID :: SynsetID
  , word :: Text
  , pos :: Char
  } deriving Show

type Hypernym = (SynsetID, SynsetID)

sc :: Parser ()
sc = L.space space1 empty empty

lexeme :: Parser a -> Parser a
lexeme = L.lexeme sc

symbol :: Text -> Parser Text
symbol = L.symbol sc

integer = lexeme L.decimal

quotedText :: Parser Text
quotedText = T.pack <$> (char '\'' *> manyTill L.charLiteral (char '\''))

synsetParser :: Parser Synset
synsetParser = do
  _ <- string "s("          
  sid <- integer            
  _ <- char ',' >> integer 
  _ <- char ',' >> space    
  w <- quotedText           
  _ <- char ',' >> space    
  p <- lowerChar            
  _ <- manyTill anySingle (char '.') 
  return (Synset sid w p)   

hypernymParser :: Parser Hypernym
hypernymParser = do
  _ <- string "hyp("       
  child <- integer          
  _ <- char ','             
  parent <- integer         
  _ <- string ")."         
  return (child, parent)    

parseSynsets :: Text -> [Synset]
parseSynsets input = case parse (many (synsetParser <* space)) "" input of
  Left err -> error (errorBundlePretty err) 
  Right xs -> filter (\s -> pos s == 'n') xs 

parseHypernyms :: Text -> [Hypernym]
parseHypernyms input = case parse (many (hypernymParser <* space)) "" input of
  Left err -> error (errorBundlePretty err) 
  Right xs -> xs

main :: IO ()
main = do
  synText <- TIO.readFile "wn_s_sample.pl"
  hypText <- TIO.readFile "wn_hyp_sample.pl"

  let synsets = parseSynsets synText
  let hypernyms = parseHypernyms hypText

  putStrLn "\nParsed Synsets:"
  mapM_ print synsets 

  putStrLn "\nParsed Hypernyms:"
  mapM_ print hypernyms 
