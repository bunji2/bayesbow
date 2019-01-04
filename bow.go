// Package bayesbow --- Bayes Bag of Words ベイズ推定を用いて文書の分類を行うパッケージ。
package bayesbow

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"strings"
)

// Bow : Bag of Words --- 文書群を単語の集合で表現する型。文書群データの型。
type Bow struct {
	Note           string              `json:"note"`           // ノート
	LabelNames     []string            `json:"labelnames"`     // ラベル名のリスト (ラベルID → ラベル名)
	LabelCount     int                 `json:"labelcount"`     // ラベルの個数。LabelCount==len(LabelNames)
	Words          []string            `json:"words"`          // 単語帳 (単語ID → 単語文字列)
	WordCount      int                 `json:"wordcount"`      // 単語数。WordCount==len(Words)
	WordDocCount   map[int]int         `json:"worddoccount"`   // 単語の出現する文書数 (単語ID → 文書数)
	DocCount       int                 `json:"doccount"`       // 文書数
	LWF            map[int]map[int]int `json:"lwf"`            // ラベルごとの単語の出現数 (ラベルID → 単語ID → 出現数)
	LabelWordCount map[int]int         `json:"labelwordcount"` // ラベルごとの単語数 (ラベルID → 単語数)
	PL             []float64           `json:"pl"`             // ラベルごとの確率。PL : Property of Label
	LabelDocCount  []int               `json:"labeldoccount"`  // ラベルごとの文書数 (ラベルID → 文書数)
	idxs           map[string]int      // 単語帳(Words集計用) (単語文字列 → 単語ID)
}

// 例 ラベルID labelID に属する文書群に出現する単語 "foo" の出現数を調べる。
//      wordID := dd.idxs["foo"]
//      numWord := dd.LWF[labelID][wordID]

// New : 文書群データの作成
func New(note string, labelNames []string) (r *Bow) {
	labelCount := len(labelNames)
	pl := make([]float64, labelCount)
	lwf := map[int]map[int]int{}
	for labelID := 0; labelID < labelCount; labelID++ {
		pl[labelID] = float64(1.0) / float64(labelCount)
		lwf[labelID] = map[int]int{}
	}

	r = &Bow{
		Note:           note,
		LWF:            lwf,
		PL:             pl,
		idxs:           map[string]int{},
		WordCount:      0,
		WordDocCount:   map[int]int{},
		LabelWordCount: map[int]int{},
		LabelNames:     labelNames,
		LabelCount:     labelCount,
		LabelDocCount:  make([]int, labelCount),
	}
	return
}

// Predict : 文書のラベルを推定する
// r --- 推定したラベルID
// pld --- 各ラベルIDごとの確率
func (dd *Bow) Predict(words []string) (r int, pld []float64) {
	// 文書内の単語の出現回数 (単語ID→単語の出現回数)
	freqWord := map[int]int{}
	for _, word := range words {

		// ストップワードは除外する
		if word == "" || conf.UseStopWords && IsStopWord(word) {
			continue
		}

		// 新しい単語かどうかを dd.idxs に登録されているかでチェック
		_, ok := dd.idxs[word]
		if !ok {
			// dd.idxs に登録されていなければ最新の単語追加し、dd.WordCountをインクリメント
			dd.idxs[word] = dd.WordCount
			dd.WordCount++
		}

		// 単語ID の取得
		wordID := dd.idxs[word]

		// 文書内の単語出現回数
		freqWord[wordID] = freqWord[wordID] + 1

	}

	// ある文書を前提とした各ラベルの確率。（イメージとしてある文書が各ラベルにどれだけ重なりがあるかを単語の出現回数を元に求める）
	pld = dd.PLD(freqWord)
	maxLabelID := 0
	for labelID := 0; labelID < dd.LabelCount; labelID++ {
		//dd.PL[labelID] = pld[labelID] // ←ベイズ更新？！
		if pld[labelID] > pld[maxLabelID] {
			maxLabelID = labelID
		}
	}
	r = maxLabelID
	return
}

// Add : 文書を追加する
func (dd *Bow) Add(id string, words []string, labels []int) {

	// サンプル
	// dd.Add("文書001", []string{"これ", "は", "ペン", "です"}, []int{34})

	// 単語リスト（単語IDのリスト）
	seq := []int{}
	// 文書内の単語の出現回数 (単語ID→単語の出現回数)
	freqWord := map[int]int{}

	for _, word := range words {

		// ストップワードは除外する
		if word == "" || conf.UseStopWords && IsStopWord(word) {
			continue
		}

		// idx --- 単語ID 各単語の全文書横断で一意な番号
		_, ok := dd.idxs[word]
		if !ok {
			// dd.idxs に登録されていなければ最新の単語追加し、dd.WordCountをインクリメント
			dd.idxs[word] = dd.WordCount
			dd.WordCount++
		}

		// 単語ID の取得
		idx := dd.idxs[word]
		// 単語リストに追加
		seq = append(seq, idx)

		// 文書内の単語出現回数
		freqWord[idx] = freqWord[idx] + 1

	}

	// 文書の総数をインクリメント
	dd.DocCount++

	// 出現した各単語IDについて、WordDocCount,LWF,LabelWordCount を更新する。
	// LWF (each Labels's Word Frequency) : ラベルごとの各単語の出現数
	for wordID := range freqWord {
		// [MEMO] freqWord は map なので、freqWord から range で引っ張っている wordID は
		// 重複がないので、以下の処理では wordID の重複を意識しないで済んでいることに注意。

		// 各ラベルごとの単語の出現数をインクリメントする
		for _, labelID := range labels {

			// 単語の出現する文書数をカウントアップする
			dd.WordDocCount[wordID] = dd.WordDocCount[wordID] + 1

			// ラベルごと単語ごとの出現数をカウントアップする
			dd.LWF[labelID][wordID] = dd.LWF[labelID][wordID] + freqWord[wordID]

			// ラベルごとの単語数をカウントアップする
			dd.LabelWordCount[labelID] = dd.LabelWordCount[labelID] + freqWord[wordID]
		}
	}

	for _, labelID := range labels {
		// 該当するラベルの文書数をカウントアップする
		dd.LabelDocCount[labelID] = dd.LabelDocCount[labelID] + 1
	}
	return
}

// WordDocCountOf : 与えられた単語の出現文書数
func (dd *Bow) WordDocCountOf(word string) int {
	idx, ok := dd.idxs[word]
	if !ok {
		return 0
	}
	return dd.WordDocCount[idx]
}

// UpdatePL : PL の更新
func (dd *Bow) UpdatePL() {
	for labelID := 0; labelID < dd.LabelCount; labelID++ {
		if dd.LabelDocCount[labelID]+1 > dd.DocCount+dd.LabelCount {
			fmt.Println("dd.LabelDocCount[", labelID, "] =", dd.LabelDocCount[labelID])
			fmt.Println("dd.DocCount =", dd.DocCount)
			fmt.Println("dd.LabelCount =", dd.LabelCount)
			panic("dd.LabelDocCount[labelID]+1 > dd.DocCount+dd.LabelCount !!")
		}
		dd.PL[labelID] = float64(dd.LabelDocCount[labelID]+1) / float64(dd.DocCount+dd.LabelCount)
	}
}

// PWL : Property of Word in Label あるラベルにおける所定の単語の出現率（log版）
func (dd *Bow) PWL(labelID, wordID int) (r float64) {

	r = math.Log(float64(dd.LWF[labelID][wordID]+1) / float64(dd.LabelWordCount[labelID]+dd.WordCount))
	return
}

// PLD : Property of Label over Document ある文書を前提としたラベルの確率（log版）
// イメージとしてある文書が各ラベルにどれだけ重なりがあるかを単語の出現回数を元に求める
func (dd *Bow) PLD(wordFreq map[int]int) (r []float64) {
	dd.UpdatePL()
	r = make([]float64, dd.LabelCount)
	sum := float64(0.0)
	for labelID := 0; labelID < dd.LabelCount; labelID++ {
		r[labelID] = math.Log(dd.PL[labelID])
		for wordID := range wordFreq {
			r[labelID] += dd.PWL(labelID, wordID)
		}
		r[labelID] = math.Exp(r[labelID])
		sum += r[labelID]
	}
	for labelID := 0; labelID < dd.LabelCount; labelID++ {
		r[labelID] /= sum
	}
	return
}

func (dd *Bow) updateWords() {
	dd.Words = make([]string, len(dd.idxs))
	for word, idx := range dd.idxs {
		dd.Words[idx] = word
	}
}

// Load : 文書データの読み出し
func Load(inFile string) (dd *Bow, err error) {
	var bytes []byte
	bytes, err = ioutil.ReadFile(inFile)
	if err != nil {
		return
	}
	var d Bow
	err = json.Unmarshal(bytes, &d)
	if err != nil {
		return
	}

	// d.idxs : 単語帳 (単語文字列 → 単語ID)
	d.idxs = map[string]int{}
	for idx, word := range d.Words {
		d.idxs[word] = idx
	}

	dd = &d
	return
}

// Save : 文書データの保存
func (dd *Bow) Save(outFile string) (err error) {

	// Words を更新
	dd.updateWords()

	var w *os.File
	w, err = os.Create(outFile)
	if err != nil {
		return
	}
	defer w.Close()
	var b []byte
	b, err = json.Marshal(dd)

	//_, err = w.Write(bytes)
	var out bytes.Buffer
	json.Indent(&out, b, "", "  ")
	out.WriteTo(w)
	return
}

// IsStopWord : ストップワードかどうか
func IsStopWord(x string) (r bool) {
	for _, word := range conf.StopWords {
		if x == word {
			r = true
			return
		}
	}
	for _, wordClass := range conf.StopWordClasses {
		if strings.HasPrefix(x, wordClass) {
			r = true
			break
		}
	}
	return
}
