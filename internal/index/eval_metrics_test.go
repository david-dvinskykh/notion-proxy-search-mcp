package index

import "testing"

func TestCanonicalPageIDAcceptsEveryShapeNotionHandsOut(t *testing.T) {
	want := "3abaf4f8da09819e9e12f82394fa6c02"
	for _, in := range []string{
		"3abaf4f8-da09-819e-9e12-f82394fa6c02",
		"3abaf4f8da09819e9e12f82394fa6c02",
		"https://app.notion.com/p/SoftServe-Poland-244-20-PLN-LuxMed-3abaf4f8da09819e9e12f82394fa6c02",
		"HTTPS://APP.NOTION.COM/P/3ABAF4F8DA09819E9E12F82394FA6C02?pvs=204",
	} {
		if got := CanonicalPageID(in); got != want {
			t.Errorf("CanonicalPageID(%q) = %q", in, got)
		}
	}
}

func TestSummariseCountsRanksAndMisses(t *testing.T) {
	m := Summarise("hybrid", []int{1, 1, 2, 5, 0, 9})
	if m.Hit1 != 2 || m.Hit3 != 3 || m.Hit5 != 4 || m.Missed != 1 {
		t.Fatalf("counts wrong: %+v", m)
	}
	// 1 + 1 + 1/2 + 1/5 + 0 + 1/9, over six cases.
	if want := (1 + 1 + 0.5 + 0.2 + 1.0/9) / 6; m.MRR < want-1e-9 || m.MRR > want+1e-9 {
		t.Fatalf("MRR = %.6f, want %.6f", m.MRR, want)
	}
}

func TestRankOfMatchesRegardlessOfIDShape(t *testing.T) {
	results := []Result{
		{PageID: "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"},
		{PageID: "3abaf4f8-da09-819e-9e12-f82394fa6c02"},
	}
	if got := RankOf(results, "https://app.notion.com/p/x-3abaf4f8da09819e9e12f82394fa6c02"); got != 2 {
		t.Fatalf("rank = %d, want 2", got)
	}
	if got := RankOf(results, "0000ffff-0000-0000-0000-000000000000"); got != 0 {
		t.Fatalf("missing page reported at rank %d", got)
	}
}
