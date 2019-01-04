package bayesbow

// Config :
type Config struct {
	UseStopWords    bool
	StopWords       []string
	StopWordClasses []string
}

func makeDefaultConfig() Config {
	return Config{}
}

// Init :
func Init(c Config) (err error) {
	conf = c

	// [TODO] なにかデフォルトでやらないと困ることがあればここに入れる。

	return
}

var conf Config
